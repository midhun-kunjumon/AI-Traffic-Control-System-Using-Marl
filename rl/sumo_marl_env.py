import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
import numpy as np
import os
import sys
import random

# --- JUNCTION CONFIGURATION (mirrors sim_runner.py) ---
JUNCTION_CONFIG = {
    "J0": {
        "N": ["JN_to_J0_0", "JN_to_J0_1"],
        "E": ["JE_to_J0_0", "JE_to_J0_1"],
        "S": ["JS_to_J0_0", "JS_to_J0_1"],
        "W": ["JW_to_J0_0", "JW_to_J0_1"]
    },
    "JN": {
        "N": ["N_to_JN_0", "N_to_JN_1"],
        "E": ["JN_E_to_JN_0", "JN_E_to_JN_1"],
        "S": ["J0_to_JN_0", "J0_to_JN_1"],
        "W": ["JN_W_to_JN_0", "JN_W_to_JN_1"]
    },
    "JS": {
        "N": ["J0_to_JS_0", "J0_to_JS_1"],
        "E": ["JS_E_to_JS_0", "JS_E_to_JS_1"],
        "S": ["S_to_JS_0", "S_to_JS_1"],
        "W": ["JS_W_to_JS_0", "JS_W_to_JS_1"]
    },
    "JW": {
        "N": ["JW_N_to_JW_0", "JW_N_to_JW_1"],
        "E": ["J0_to_JW_0", "J0_to_JW_1"],
        "S": ["JW_S_to_JW_0", "JW_S_to_JW_1"],
        "W": ["W_to_JW_0", "W_to_JW_1"]
    },
    "JE": {
        "N": ["JE_N_to_JE_0", "JE_N_to_JE_1"],
        "E": ["E_to_JE_0", "E_to_JE_1"],
        "S": [],
        "W": ["J0_to_JE_0", "J0_to_JE_1"]
    }
}

ALL_JUNCTIONS = ["J0", "JN", "JS", "JW", "JE"]


class SumoMarlEnv(gym.Env):
    """
    Improved MARL Training Environment with:
    - 13-dim observation (phase + 4 queues + 4 counts + 4 waiting times)
    - Better reward shaping (waiting + queue + throughput + switch penalty)
    - Random junction selection per episode for parameter sharing
    - Collision-safe yellow transitions
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, use_gui=False, num_seconds=20000,
                 decision_interval=5.0, min_green=10.0, yellow_duration=3.0):
        super(SumoMarlEnv, self).__init__()

        self._cfg_file = cfg_file
        self._gui = use_gui
        self._max_steps = num_seconds
        self._decision_interval = decision_interval
        self._min_green = min_green
        self._yellow_duration = yellow_duration

        self._step = 0
        self._sim_step = 0
        self._tls_id = "J0"  # Will be randomized on reset
        self._last_switch_step = 0
        self._last_phase = 0
        self._num_phases = 8
        self._prev_departed = 0

        # Action Space: 0 = Keep, 1 = Switch
        self.action_space = spaces.Discrete(2)

        # Observation Space (Size 13):
        # [Phase, Q_N, Q_E, Q_S, Q_W, C_N, C_E, C_S, C_W, W_N, W_E, W_S, W_W]
        self.observation_space = spaces.Box(low=0, high=500, shape=(13,), dtype=np.float32)

        if self._gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

    def _get_sumo_cmd(self):
        return [
            self._sumo_binary,
            "-c", self._cfg_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",  # Disable teleportation
            "--collision.check-junctions", "true",
            "--collision.action", "warn",
            "--start", "--quit-on-end"
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            traci.close()
        except:
            pass

        traci.start(self._get_sumo_cmd())
        self._step = 0
        self._sim_step = 0
        self._last_switch_step = 0
        self._prev_departed = 0

        # Randomly pick a junction for this episode (MARL parameter sharing)
        self._tls_id = random.choice(ALL_JUNCTIONS)
        self._config = JUNCTION_CONFIG[self._tls_id]

        # Set up phase count
        if self._tls_id == "JE":
            self._num_phases = 6
        else:
            self._num_phases = 8

        self._last_phase = traci.trafficlight.getPhase(self._tls_id)

        return self._get_observation(), {}

    def step(self, action):
        self._step += 1
        current_phase = traci.trafficlight.getPhase(self._tls_id)
        dt = traci.simulation.getDeltaT()

        # Time since last phase switch (in simulation seconds)
        time_since_switch = (self._sim_step - self._last_switch_step) * dt

        switched = False

        # Enforce minimum green time
        if action == 1 and time_since_switch < self._min_green:
            action = 0

        if action == 1:
            # --- COLLISION-SAFE SWITCH: Green -> Yellow -> Next Green ---
            # 1. Set Yellow
            yellow_phase = (current_phase + 1) % self._num_phases
            traci.trafficlight.setPhase(self._tls_id, yellow_phase)

            # 2. Simulate Yellow Duration
            yellow_steps = max(1, int(self._yellow_duration / dt))
            for _ in range(yellow_steps):
                traci.simulationStep()
                self._sim_step += 1
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break

            # 3. Switch to Next Green
            next_green = (yellow_phase + 1) % self._num_phases
            traci.trafficlight.setPhase(self._tls_id, next_green)
            self._last_switch_step = self._sim_step
            switched = True

        # Run simulation for decision interval
        reward = 0.0
        interval_steps = max(1, int(self._decision_interval / dt))

        departed_before = traci.simulation.getDepartedNumber()

        for _ in range(interval_steps):
            traci.simulationStep()
            self._sim_step += 1
            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        departed_after = traci.simulation.getDepartedNumber()

        # --- REWARD COMPUTATION ---
        all_lanes = []
        for direction in ["N", "E", "S", "W"]:
            all_lanes.extend(self._config[direction])

        total_waiting = sum([traci.lane.getWaitingTime(l) for l in all_lanes if l])
        total_queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in all_lanes if l])
        throughput = departed_after - departed_before

        reward = -(total_waiting) - 0.3 * total_queue + 0.1 * max(0, throughput)

        # Penalize unnecessary switching (anti-jitter)
        if switched:
            reward -= 2.0

        obs = self._get_observation()
        done = traci.simulation.getMinExpectedNumber() <= 0 or self._step >= self._max_steps
        truncated = False

        return obs, reward, done, truncated, {}

    def _get_observation(self):
        phase = traci.trafficlight.getPhase(self._tls_id)

        queues = []
        counts = []
        waits = []

        for direction in ["N", "E", "S", "W"]:
            lanes = self._config[direction]
            if not lanes:
                queues.append(0.0)
                counts.append(0.0)
                waits.append(0.0)
            else:
                q = sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
                c = sum([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
                w = sum([traci.lane.getWaitingTime(l) for l in lanes])
                queues.append(float(q))
                counts.append(float(c))
                waits.append(float(w))

        obs = np.array([phase] + queues + counts + waits, dtype=np.float32)
        return obs

    def close(self):
        try:
            traci.close()
        except:
            pass
