import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
import numpy as np
import os
import sys

class SumoIntersectionEnv(gym.Env):
    """
    Custom Gymnasium Environment for a single Intersection Simulation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, net_file, route_file, use_gui=False, num_seconds=20000, use_sumocfg=False):
        super(SumoIntersectionEnv, self).__init__()
        
        self._net = net_file
        self._route = route_file
        self._gui = use_gui
        self._max_steps = num_seconds
        self._use_sumocfg = use_sumocfg

        self._step = 0
        self._tls_id = "J0" # As identified in net.xml
        
        # Stability Params
        self._min_green = 10.0 # Minimum Green Time in seconds
        self._last_switch_step = 0 # Step index of last switch

        # Define Action Space:
        # 0: Keep current phase
        # 1: Switch to next phase
        self.action_space = spaces.Discrete(2)

        # Define Observation Space (Size 9):
        # [Phase, Q_N, Q_E, Q_S, Q_W, C_N, C_E, C_S, C_W]
        # We aggregate the 2 lanes per arm into 1 value.
        self.observation_space = spaces.Box(low=0, high=100, shape=(9,), dtype=np.float32)

        # Incoming Lanes map (Lane 0 and Lane 1)
        self._incoming_lanes = [
            # North (Coming from JN)
            "JN_to_J0_0", "JN_to_J0_1",
            # East (Coming from JE)
            "JE_to_J0_0", "JE_to_J0_1",
            # South (Coming from JS)
            "JS_to_J0_0", "JS_to_J0_1",
            # West (Coming from JW)
            "JW_to_J0_0", "JW_to_J0_1"
        ]
        
        if self._gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        if self._use_sumocfg:
             self.sumo_cmd = [
                 self._sumo_binary,
                 "-c", self._net, # passing cfg as net argument convenience or separate
                 "--no-step-log", "true",
                 "--waiting-time-memory", "1000"
            ]
        else:
            self.sumo_cmd = [
                 self._sumo_binary,
                 "-n", self._net,
                 "-r", self._route,
                 "--no-step-log", "true",
                 "--waiting-time-memory", "1000"
            ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            traci.close()
        except:
            pass

        traci.start(self.sumo_cmd)
        self._step = 0
        
        return self._get_observation(), {}

    def step(self, action):
        self._step += 1
        current_phase = traci.trafficlight.getPhase(self._tls_id)
        
        # Stability Constraint: Min Green Time
        # Only allow switching if enough time passed since last switch (or since start)
        # We assume simulation step is 0.45s 
        time_since_last_switch = (self._step - self._last_switch_step) * 0.45
        
        if action == 1 and time_since_last_switch < self._min_green:
            # FORCE KEEP PHASE
            action = 0 
        
        # Action 1: Switch Phase
        # Logic: Green -> Yellow (Fixed 3s) -> Next Green
        if action == 1:
            # Assume we are in a Green phase (even index: 0, 2, 4, 6)
            # If we happen to be in yellow, we just continue.
            
            # 1. Switch to Yellow
            next_yellow = (current_phase + 1) % 8
            traci.trafficlight.setPhase(self._tls_id, next_yellow)
            
            # 2. Simulate Yellow Duration (e.g., 3 seconds)
            # Yellow phases in our add.xml are 3s.
            steps_yellow = int(3.0 / 0.45) # approx 6-7 steps
            if steps_yellow < 1: steps_yellow = 1
            
            for _ in range(steps_yellow):
                traci.simulationStep()
                
            # 3. Switch to Next Green
            next_green = (next_yellow + 1) % 8
            traci.trafficlight.setPhase(self._tls_id, next_green)
            
            # UPDATE LAST SWITCH TIMESTAMP
            # We add steps_yellow to _step?
            # self._step is agent steps. But simulation steps advanced.
            # Let's adjust self._step logic or just use a dedicated sim_step counter.
            # Simplest: Just set last_switch_step to current self._step
            # But wait, self._step increments by 1 per agent act.
            # Time calculation above relies on steps.
            # Agent acts every 5s sim time?
            # In __init__: self._min_green = 10.0
            self._last_switch_step = self._step 
        
        # Action 0: Keep Phase (Do nothing, stay Green)
        
        # Run simulation for decision interval (e.g. 5 seconds)
        # Note: If we switched, we already consumed 3s. We should run a bit more or count it?
        # Let's run a fixed 5s interval for the agent's "Clean Green" time.
        
        reward = 0
        steps_interval = int(5.0 / 0.45) # approx 11 steps
        
        for _ in range(steps_interval): 
            traci.simulationStep()
            reward += self._compute_reward()
            if traci.simulation.getMinExpectedNumber() <= 0:
                 break
        
        obs = self._get_observation()
        done = traci.simulation.getMinExpectedNumber() <= 0 or self._step >= self._max_steps
        truncated = False
        info = {}

        return obs, reward, done, truncated, info

    def _get_observation(self):
        phase = traci.trafficlight.getPhase(self._tls_id)
        
        # Aggregate per Arm (2 lanes each)
        # 0,1 -> North
        # 2,3 -> East
        # 4,5 -> South
        # 6,7 -> West
        
        queues = []
        counts = []
        
        # North
        queues.append(traci.lane.getLastStepHaltingNumber("JN_to_J0_0") + traci.lane.getLastStepHaltingNumber("JN_to_J0_1"))
        counts.append(traci.lane.getLastStepVehicleNumber("JN_to_J0_0") + traci.lane.getLastStepVehicleNumber("JN_to_J0_1"))
        
        # East
        queues.append(traci.lane.getLastStepHaltingNumber("JE_to_J0_0") + traci.lane.getLastStepHaltingNumber("JE_to_J0_1"))
        counts.append(traci.lane.getLastStepVehicleNumber("JE_to_J0_0") + traci.lane.getLastStepVehicleNumber("JE_to_J0_1"))

        # South
        queues.append(traci.lane.getLastStepHaltingNumber("JS_to_J0_0") + traci.lane.getLastStepHaltingNumber("JS_to_J0_1"))
        counts.append(traci.lane.getLastStepVehicleNumber("JS_to_J0_0") + traci.lane.getLastStepVehicleNumber("JS_to_J0_1"))

        # West
        queues.append(traci.lane.getLastStepHaltingNumber("JW_to_J0_0") + traci.lane.getLastStepHaltingNumber("JW_to_J0_1"))
        counts.append(traci.lane.getLastStepVehicleNumber("JW_to_J0_0") + traci.lane.getLastStepVehicleNumber("JW_to_J0_1"))
            
        obs = np.array([phase] + queues + counts, dtype=np.float32)
        return obs

    def _compute_reward(self):
        # Reward = Negative (Waiting Time + 0.5 * Queue Length)
        total_waiting_time = sum([traci.lane.getWaitingTime(lane) for lane in self._incoming_lanes])
        total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self._incoming_lanes])
        reward = - (total_waiting_time + 0.5 * total_queue)
        return reward

    def close(self):
        traci.close()
