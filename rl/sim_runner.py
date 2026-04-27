import os
import sys
import traci
import sumolib
import numpy as np
import json
from stable_baselines3 import PPO

# Add siblings to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traci"))
from emergency_manager import EmergencyManager
import xml.etree.ElementTree as ET

# Global Results Cache (or write to file)
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_results.json")

# --- JUNCTION CONFIGURATION ---
# Mapping 4-Way Model Inputs to specific Lanes for each Junction
# Inputs: North, East, South, West (2 lanes each)
JUNCTION_CONFIG = {
    "J0": {
        "N": ["JN_to_J0_0", "JN_to_J0_1"],
        "E": ["JE_to_J0_0", "JE_to_J0_1"],
        "S": ["JS_to_J0_0", "JS_to_J0_1"],
        "W": ["JW_to_J0_0", "JW_to_J0_1"]
    },
    "JN": {
        "N": ["N_to_JN_0", "N_to_JN_1"],    # From World North
        "E": ["JN_E_to_JN_0", "JN_E_to_JN_1"],
        "S": ["J0_to_JN_0", "J0_to_JN_1"],   # From Center J0
        "W": ["JN_W_to_JN_0", "JN_W_to_JN_1"]
    },
    "JS": {
        "N": ["J0_to_JS_0", "J0_to_JS_1"],   # From Center J0
        "E": ["JS_E_to_JS_0", "JS_E_to_JS_1"],
        "S": ["S_to_JS_0", "S_to_JS_1"],     # From World South
        "W": ["JS_W_to_JS_0", "JS_W_to_JS_1"]
    },
    "JW": {
        "N": ["JW_N_to_JW_0", "JW_N_to_JW_1"],
        "E": ["J0_to_JW_0", "J0_to_JW_1"],   # From Center J0
        "S": ["JW_S_to_JW_0", "JW_S_to_JW_1"],
        "W": ["W_to_JW_0", "W_to_JW_1"]     # From World West
    },
    "JE": {
        "N": ["JE_N_to_JE_0", "JE_N_to_JE_1"],
        "E": ["E_to_JE_0", "E_to_JE_1"],     # From World East
        "S": [],                             # Virtual South (Empty)
        "W": ["J0_to_JE_0", "J0_to_JE_1"]    # From Center J0
    }
}

ALL_JUNCTIONS = ["J0", "JN", "JS", "JW", "JE"]


class TrafficAgent:
    """
    Traffic agent controlling one junction using the trained RL model.
    
    Phase transition is ALWAYS 3-stage for safety:
      Current Green -> Yellow (5s) -> ALL-RED clearance (3s) -> Next Green
    
    The all-red clearance ensures vehicles already in the junction
    have time to fully clear before the conflicting green starts.
    """
    
    STATE_IDLE = "IDLE"
    STATE_YELLOW = "YELLOW"
    STATE_ALL_RED = "ALL_RED"

    def __init__(self, tls_id, model, emg_manager, enable_emergency=False):
        self.tls_id = tls_id
        self.model = model
        self.emg_manager = emg_manager
        self.enable_emergency = enable_emergency
        self.phase_start_step = 0
        self.last_switch_step = 0
        
        # 3-Stage State Machine
        self.switch_state = self.STATE_IDLE
        self.switch_start_step = 0
        self.clearance_start_step = 0
        self.target_green_phase = 0
        
        # Timing (seconds)
        self.yellow_duration = 5.0      # Long enough for vehicles at 50 km/h to stop
        self.all_red_duration = 3.0     # Clearance for vehicles already in junction
        self.min_green = 15.0           # Minimum green before any switch allowed
        
        self.config = JUNCTION_CONFIG.get(tls_id)
        if not self.config:
            raise ValueError(f"Unknown junction ID: {tls_id}")
            
        # Phase mapping: Even = green, Odd = yellow
        if self.tls_id == "JE":
            self.num_phases = 6
            self.dir_map = {"N": 0, "E": 2, "W": 4, "S": 0}
        else:
            self.num_phases = 8
            self.dir_map = {"N": 0, "E": 2, "S": 4, "W": 6}
        
        # Number of signal heads (set on first use)
        self._num_signals = None
            
        # RL Control Frequency
        self.last_action_step = 0
        self.action_interval = 5.0

    def _is_green_phase(self, phase):
        """Even phases are green, odd phases are yellow."""
        return phase % 2 == 0

    def _get_yellow_for_green(self, green_phase):
        """Get the yellow phase index that follows a green phase."""
        return (green_phase + 1) % self.num_phases

    def _set_all_red(self):
        """Set ALL signals at this junction to RED for safe clearance."""
        if self._num_signals is None:
            state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            self._num_signals = len(state)
        traci.trafficlight.setRedYellowGreenState(self.tls_id, 'r' * self._num_signals)

    def act(self, step):
        dt = traci.simulation.getDeltaT()
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        
        # --- 1. EMERGENCY OVERRIDE (Top Priority - Every Step) ---
        if self.enable_emergency:
            ev_info = self._check_local_emergency()
            if ev_info:
                ev_lane, ev_direction = ev_info
                target_phase = self.dir_map.get(ev_direction, 0)
                 
                if self._is_green_phase(current_phase) and current_phase == target_phase and self.switch_state == self.STATE_IDLE:
                    # Already green for EV direction - hold it
                    self.phase_start_step = step
                    self.last_action_step = step
                elif self.switch_state != self.STATE_IDLE and self.target_green_phase == target_phase:
                    # Already transitioning to correct phase
                    self._update_state_machine(step, dt)
                    self.last_action_step = step
                elif self.switch_state != self.STATE_IDLE:
                    # Transitioning to wrong phase - redirect target
                    self.target_green_phase = target_phase
                    self._update_state_machine(step, dt)
                    self.last_action_step = step
                else:
                    # Start transition for EV
                    self._begin_transition(target_phase, current_phase, step)
                    self.last_action_step = step
                return

        # --- 2. ANTI-STARVATION FAILSAFE (> 120s wait) ---
        starved_lane = self._check_starvation()
        if starved_lane:
            target_phase = self._get_green_phase_for_lane(starved_lane)
             
            if self._is_green_phase(current_phase) and current_phase == target_phase and self.switch_state == self.STATE_IDLE:
                self.phase_start_step = step  # Hold green
            elif self.switch_state != self.STATE_IDLE and self.target_green_phase == target_phase:
                pass  # Already going there
            elif self.switch_state == self.STATE_IDLE:
                self._begin_transition(target_phase, current_phase, step)
             
            if self.switch_state != self.STATE_IDLE:
                self._update_state_machine(step, dt)
            return

        # --- 3. STATE MACHINE UPDATE (Yellow / All-Red transitions) ---
        if self.switch_state != self.STATE_IDLE:
            self._update_state_machine(step, dt)
            return
            
        # --- 4. RL CONTROL (Normal Operation) ---
        time_since_last_action = (step - self.last_action_step) * dt
        if time_since_last_action < self.action_interval:
            return
            
        self.last_action_step = step
        
        # Only act from green phases
        if not self._is_green_phase(current_phase):
            return
        
        obs = self._get_observation(current_phase)
        action, _ = self.model.predict(obs, deterministic=True)
        
        time_since_change = (step - self.phase_start_step) * dt
        
        if action == 1 and time_since_change >= self.min_green:
            self._initiate_smart_switch(current_phase, step)

    def _update_state_machine(self, step, dt):
        """
        3-stage transition ensures vehicles clear the junction:
        
          YELLOW  --[5s]--> ALL_RED  --[3s]--> target GREEN
          
        No conflicting green is ever set while vehicles could be
        in the junction from the previous direction.
        """
        if self.switch_state == self.STATE_YELLOW:
            elapsed = (step - self.switch_start_step) * dt
            if elapsed >= self.yellow_duration:
                # Yellow done -> ALL RED clearance
                self._set_all_red()
                self.switch_state = self.STATE_ALL_RED
                self.clearance_start_step = step
                
        elif self.switch_state == self.STATE_ALL_RED:
            elapsed = (step - self.clearance_start_step) * dt
            if elapsed >= self.all_red_duration:
                # Clearance done -> safe to set next green
                # Restore the split_phase program before setting the phase index
                # since _set_all_red (which uses setRedYellowGreenState) overrides the program.
                traci.trafficlight.setProgram(self.tls_id, "split_phase")
                traci.trafficlight.setPhase(self.tls_id, self.target_green_phase)
                self.switch_state = self.STATE_IDLE
                self.phase_start_step = step
                self.last_switch_step = step
                self.last_action_step = step

    def _begin_transition(self, target_phase, current_phase, step):
        """
        Start the 3-stage transition: Yellow -> All-Red -> Green.
        If already in a non-green phase, skip to all-red directly.
        """
        if self._is_green_phase(current_phase):
            # Green -> set yellow for current direction
            yellow = self._get_yellow_for_green(current_phase)
            traci.trafficlight.setPhase(self.tls_id, yellow)
            self.switch_state = self.STATE_YELLOW
            self.switch_start_step = step
        else:
            # Already in yellow or other -> go to all-red immediately
            self._set_all_red()
            self.switch_state = self.STATE_ALL_RED
            self.clearance_start_step = step
            self.switch_start_step = step
        
        self.target_green_phase = target_phase

    def _check_starvation(self):
        """Check all incoming lanes for excessive waiting (> 120s)."""
        max_wait = 0
        worst_lane = None
        for direction in ["N", "E", "S", "W"]:
            for lane in self.config[direction]:
                if not lane:
                    continue
                wait = traci.lane.getWaitingTime(lane)
                if wait > 120.0 and wait > max_wait:
                    max_wait = wait
                    worst_lane = lane
        return worst_lane

    def _check_local_emergency(self):
        """Scan incoming lanes AND upstream edges for emergency vehicles.
        Returns (lane, direction) or None."""
        for direction in ["N", "E", "S", "W"]:
            lanes = self.config[direction]
            for lane in lanes:
                if not lane:
                    continue
                # Check vehicles on the lane
                try:
                    vehs = traci.lane.getLastStepVehicleIDs(lane)
                    for veh in vehs:
                        try:
                            vtype = traci.vehicle.getTypeID(veh)
                            vclass = traci.vehicle.getVehicleClass(veh)
                            if vtype == "emergency" or vclass == "emergency":
                                return (lane, direction)
                        except traci.TraCIException:
                            continue
                except traci.TraCIException:
                    continue
                
                # Check upstream edge for approaching emergency vehicles
                edge_id = lane.rsplit("_", 1)[0]
                try:
                    edge_vehs = traci.edge.getLastStepVehicleIDs(edge_id)
                    for veh in edge_vehs:
                        try:
                            vtype = traci.vehicle.getTypeID(veh)
                            vclass = traci.vehicle.getVehicleClass(veh)
                            if vtype == "emergency" or vclass == "emergency":
                                return (lane, direction)
                        except traci.TraCIException:
                            continue
                except traci.TraCIException:
                    continue
        return None

    def _get_green_phase_for_lane(self, lane):
        """Get the green phase for the direction this lane belongs to."""
        for direction, lanes in self.config.items():
            if lane in lanes:
                return self.dir_map.get(direction, 0)
        return 0

    def _get_observation(self, phase):
        """Build 22-dim normalized observation matching v3 training env."""
        dt = traci.simulation.getDeltaT()
        current_step = traci.simulation.getTime() / dt if dt > 0 else 0
        phase_duration = (current_step - self.last_switch_step) * dt

        phase_norm = phase / max(1, self.num_phases - 1)
        phase_dur_norm = min(phase_duration / 120.0, 1.0)

        queues, counts, waits, densities, speeds = [], [], [], [], []

        for direction in ["N", "E", "S", "W"]:
            lanes = self.config[direction]
            if not lanes:
                queues.append(0.0); counts.append(0.0); waits.append(0.0)
                densities.append(0.0); speeds.append(0.0)
            else:
                q = sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
                c = sum([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
                w = sum([traci.lane.getWaitingTime(l) for l in lanes])
                total_length = sum([traci.lane.getLength(l) for l in lanes])
                d = c / max(1.0, total_length) if total_length > 0 else 0.0
                s_vals = [traci.lane.getLastStepMeanSpeed(l) for l in lanes]
                s = sum(s_vals) / len(s_vals) if s_vals else 0.0
                queues.append(float(q)); counts.append(float(c)); waits.append(float(w))
                densities.append(float(d)); speeds.append(float(s))

        norm_queues = [min(q / 30.0, 1.0) for q in queues]
        norm_counts = [min(c / 40.0, 1.0) for c in counts]
        norm_waits = [min(w / 500.0, 1.0) for w in waits]
        norm_densities = [min(d / 0.15, 1.0) for d in densities]
        norm_speeds = [min(s / 15.0, 1.0) for s in speeds]

        return np.array(
            [phase_norm, phase_dur_norm] +
            norm_queues + norm_counts + norm_waits +
            norm_densities + norm_speeds,
            dtype=np.float32
        )

    def _initiate_smart_switch(self, current_phase, step):
        """Switch to the direction with highest queue (density-based)."""
        if not self._is_green_phase(current_phase):
            return
            
        queues = []
        for direction in ["N", "E", "S", "W"]:
            lanes = self.config[direction]
            if not lanes:
                queues.append(0)
            else:
                queues.append(sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes]))
            
        # Exclude current direction
        current_idx = -1
        if self.tls_id == "JE":
            if current_phase == 0: current_idx = 0
            elif current_phase == 2: current_idx = 1
            elif current_phase == 4: current_idx = 3
        else:
            current_idx = int(current_phase / 2)

        if 0 <= current_idx < len(queues): 
            queues[current_idx] = -1 
        
        best_idx = int(np.argmax(queues))
        
        if queues[best_idx] <= 0:
            # No queues - cycle to next direction
            possibles = [0, 1, 2, 3]
            if self.tls_id == "JE":
                possibles = [0, 1, 3]
            try:
                curr_loc = possibles.index(current_idx)
                best_idx = possibles[(curr_loc + 1) % len(possibles)]
            except ValueError:
                best_idx = 0

        if self.tls_id == "JE":
            phase_map = {0: 0, 1: 2, 3: 4}
            next_green = phase_map.get(best_idx, 0)
        else:
            next_green = best_idx * 2
            
        # Start 3-stage transition: Yellow -> All-Red -> Green
        self._begin_transition(next_green, current_phase, step)


def save_result(scenario_name, waiting_time, vehicle_count):
    data = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
        except: pass
    
    data[scenario_name] = {
        "waiting_time": round(waiting_time, 2),
        "vehicle_count": vehicle_count
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def parse_stats(trip_file):
    if not os.path.exists(trip_file):
        return 0, 0.0
    try:
        root = ET.parse(trip_file).getroot()
        trips = root.findall('tripinfo')
        if not trips: return 0, 0.0
        avg_wait = sum([float(t.get('waitingTime')) for t in trips]) / len(trips)
        return len(trips), avg_wait
    except:
        return 0, 0.0

def run_fixed(steps=1000, headless=False, collision_check=False, log_file=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file = os.path.join(base_path, "tripinfo_web_fixed.xml")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    
    sumo_binary = sumolib.checkBinary('sumo') if headless else sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary, "-c", cfg_file, "--tripinfo-output", trip_file,
        "--no-step-log", "true", "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--collision.check-junctions", "true", "--collision.action", "warn",
        "-e", str(steps), "--start", "--quit-on-end"
    ]
    if collision_check:
        cmd.extend(["--collision.check-junctions", "true", "--collision.action", "warn"])
    if log_file:
         cmd.extend(["--log", log_file])
    
    traci.start(cmd)
    step = 0
    while step < steps:
        traci.simulationStep()
        step += 1
        if traci.simulation.getMinExpectedNumber() <= 0: break
            
    traci.close()
    
    n, wait = parse_stats(trip_file)
    save_result("Fixed Time", wait, n)
    return n, wait

def run_stable_rl(steps=1000, enable_emergency=True, headless=False, collision_check=False, log_file=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file_name = "tripinfo_web_emg.xml" if enable_emergency else "tripinfo_web_rl.xml"
    trip_file = os.path.join(base_path, trip_file_name)
    model_path = os.path.join(base_path, "models", "ppo_marl_v3")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    model = PPO.load(model_path)
    emg_manager = EmergencyManager(tls_id="J0")
    
    sumo_binary = sumolib.checkBinary('sumo') if headless else sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary, "-c", cfg_file, "--tripinfo-output", trip_file,
        "--no-step-log", "true", "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--collision.check-junctions", "true", "--collision.action", "warn",
        "-e", str(steps), "--start", "--quit-on-end"
    ]
    
    if collision_check:
        cmd.extend(["--collision.check-junctions", "true", "--collision.action", "warn"])
    if log_file:
         cmd.extend(["--log", log_file])
    
    traci.start(cmd)
    
    # Initialize Agents for all 5 junctions
    agents = {}
    for j_id in ALL_JUNCTIONS:
        agents[j_id] = TrafficAgent(j_id, model, emg_manager, enable_emergency=enable_emergency)
        
    step = 0
    
    while step < steps:
        if traci.simulation.getMinExpectedNumber() <= 0: break

        for agent in agents.values():
            agent.act(step)

        traci.simulationStep()
        step += 1
            
    traci.close()
    
    n, wait = parse_stats(trip_file)
    scenario_name = "Emergency RL" if enable_emergency else "Stable RL"
    save_result(scenario_name, wait, n)
    return n, wait

def run_emergency_rl(steps=1000, headless=False, collision_check=False, log_file=None):
    return run_stable_rl(steps, enable_emergency=True, headless=headless, collision_check=collision_check, log_file=log_file)
