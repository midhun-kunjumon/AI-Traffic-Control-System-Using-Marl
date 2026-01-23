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

def run_fixed(steps=1000):
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file = os.path.join(base_path, "tripinfo_web_fixed.xml")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    
    sumo_binary = sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary, "-c", cfg_file, "--tripinfo-output", trip_file,
        "--no-step-log", "true", "--waiting-time-memory", "1000",
        "-e", str(steps), "--start", "--quit-on-end"
    ]
    
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

def run_stable_rl(steps=1000):
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file = os.path.join(base_path, "tripinfo_web_rl.xml")
    model_path = os.path.join(base_path, "models", "ppo_stable")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    model = PPO.load(model_path)
    emg_manager = EmergencyManager(tls_id="J0")
    
    sumo_binary = sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary, "-c", cfg_file, "--tripinfo-output", trip_file,
        "--no-step-log", "true", "--waiting-time-memory", "1000",
        "-e", str(steps), "--start", "--quit-on-end"
    ]
    
    traci.start(cmd)
    
    tls_id = "J0"
    step = 0
    phase_start_step = 0
    
    while step < steps:
        if traci.simulation.getMinExpectedNumber() <= 0: break

        # 1. Check Emergency First
        ev_lane = emg_manager.check_emergency()
        
        if ev_lane:
            # EMERGENCY OVERRIDE
            print(f"[EMERGENCY DETECTED] Lane: {ev_lane} - OVERRIDING SIGNAL")
            target_phase = emg_manager.get_override_phase(ev_lane)
            current_phase = traci.trafficlight.getPhase(tls_id)
            
            if current_phase != target_phase:
                 # Force Green for responsiveness
                traci.trafficlight.setPhase(tls_id, target_phase)
                phase_start_step = step # Reset timer
        else:
            # 2. Normal Stable RL Logic
            phase = traci.trafficlight.getPhase(tls_id)
            lane_data = _get_lane_data()
            obs = np.array([phase] + lane_data, dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            
            # Stability Logic
            dt = traci.simulation.getDeltaT()
            time_since_change = (step - phase_start_step) * dt
            min_green = 5.0
            
            if action == 1 and time_since_change >= min_green:
                 _smart_switch(tls_id)
                 phase_start_step = step

        traci.simulationStep()
        step += 1
            
    traci.close()
    
    n, wait = parse_stats(trip_file)
    save_result("Stable RL", wait, n)
    return n, wait

def run_emergency_rl(steps=1000):
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file = os.path.join(base_path, "tripinfo_web_emg.xml")
    model_path = os.path.join(base_path, "models", "ppo_stable")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    model = PPO.load(model_path)
    emg_manager = EmergencyManager(tls_id="J0")
    
    sumo_binary = sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary, "-c", cfg_file, "--tripinfo-output", trip_file,
        "--no-step-log", "true", "--waiting-time-memory", "1000",
        "-e", str(steps), "--start", "--quit-on-end"
    ]
    
    traci.start(cmd)
    
    tls_id = "J0"
    step = 0
    phase_start_step = 0
    
    while step < steps:
        if traci.simulation.getMinExpectedNumber() <= 0: break
        
        # 1. Check Emergency First
        ev_lane = emg_manager.check_emergency()
        
        if ev_lane:
            # EMERGENCY OVERRIDE
            print(f"[EMERGENCY DETECTED] Lane: {ev_lane} - OVERRIDING SIGNAL")
            target_phase = emg_manager.get_override_phase(ev_lane)
            current_phase = traci.trafficlight.getPhase(tls_id)
            
            if current_phase != target_phase:
                # Switch immediately (Yellow then Green) - or just Force Green for simplicity in Emergency?
                # Realistically, we need yellow. But for Emergency, we want FAST.
                # Let's do a fast switch: set to target directly (unsafe but effective for demo)
                # OR proper yellow. Let's do proper yellow.
                
                # If currently green for someone else -> Yellow -> Target Green
                # If currently yellow -> wait -> Target Green
                # For demo simplicity: Force Green if not already
                traci.trafficlight.setPhase(tls_id, target_phase)
                phase_start_step = step # Reset timer
            else:
                # Already green, KEEP IT GREEN
                pass
                
        else:
            # 2. Normal Stable RL Logic
            phase = traci.trafficlight.getPhase(tls_id)
            lane_data = _get_lane_data()
            obs = np.array([phase] + lane_data, dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            
            # Stability Logic
            dt = traci.simulation.getDeltaT()
            time_since_change = (step - phase_start_step) * dt
            min_green = 5.0
            
            if action == 1 and time_since_change >= min_green:
                 _smart_switch(tls_id)
                 phase_start_step = step

        traci.simulationStep()
        step += 1
            
    traci.close()
    
    n, wait = parse_stats(trip_file)
    save_result("Emergency RL", wait, n)
    return n, wait

def _get_lane_data():
    lanes = [
        ("N_to_J0_0", "N_to_J0_1"), ("E_to_J0_0", "E_to_J0_1"),
        ("S_to_J0_0", "S_to_J0_1"), ("W_to_J0_0", "W_to_J0_1")
    ]
    queues = []
    counts = []
    for l0, l1 in lanes:
        queues.append(traci.lane.getLastStepHaltingNumber(l0) + traci.lane.getLastStepHaltingNumber(l1))
        counts.append(traci.lane.getLastStepVehicleNumber(l0) + traci.lane.getLastStepVehicleNumber(l1))
    return queues + counts

def _smart_switch(tls_id):
    dt = traci.simulation.getDeltaT()
    current = traci.trafficlight.getPhase(tls_id)
    yellow_phase = (current + 1) % 8
    traci.trafficlight.setPhase(tls_id, yellow_phase)
    
    yellow_steps = int(3.0 / dt)
    for _ in range(yellow_steps):
        traci.simulationStep()
    
    # Greedy Selection
    lanes = [
        ("N_to_J0_0", "N_to_J0_1"), ("E_to_J0_0", "E_to_J0_1"),
        ("S_to_J0_0", "S_to_J0_1"), ("W_to_J0_0", "W_to_J0_1")
    ]
    lane_queues = []
    for l0, l1 in lanes:
        q = traci.lane.getLastStepHaltingNumber(l0) + traci.lane.getLastStepHaltingNumber(l1)
        lane_queues.append(q)
    
    current_lane_idx = int(current / 2)
    lane_queues[current_lane_idx] = -1 # Mask current
    
    best_idx = np.argmax(lane_queues)
    if lane_queues[best_idx] <= 0:
        best_idx = (current_lane_idx + 1) % 4
        
    next_green = best_idx * 2
    traci.trafficlight.setPhase(tls_id, next_green)
