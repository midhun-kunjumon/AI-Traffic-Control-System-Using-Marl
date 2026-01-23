
import os
import sys
import traci
import sumolib
import numpy as np
from stable_baselines3 import PPO
from sumo_env import SumoIntersectionEnv
import xml.etree.ElementTree as ET

def run_simulation(model=None, steps=1000, label="Fixed Time"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    
    trip_file = f"tripinfo_{label.lower().replace(' ', '_')}.xml"
    
    sumo_binary = sumolib.checkBinary('sumo')
    cmd = [
        sumo_binary,
        "-c", cfg_file,
        "--tripinfo-output", trip_file,
        "--no-step-log", "true",
        "-e", str(steps)
    ]
    
    traci.start(cmd)
    
    # RL Setup
    tls_id = "J0"
    
    total_waiting_time = 0
    total_cars = 0
    
    step = 0
    while step < steps:
        if model:
            # RL Control
            # Get Observation
            phase = traci.trafficlight.getPhase(tls_id)
            # Aggregate per Arm (2 lanes each)
            queues = []
            counts = []
            # N, E, S, W (same order as Env)
            lanes = [
                ("N_to_J0_0", "N_to_J0_1"),
                ("E_to_J0_0", "E_to_J0_1"),
                ("S_to_J0_0", "S_to_J0_1"),
                ("W_to_J0_0", "W_to_J0_1")
            ]
            for l0, l1 in lanes:
                queues.append(traci.lane.getLastStepHaltingNumber(l0) + traci.lane.getLastStepHaltingNumber(l1))
                counts.append(traci.lane.getLastStepVehicleNumber(l0) + traci.lane.getLastStepVehicleNumber(l1))
            
            obs = np.array([phase] + queues + counts, dtype=np.float32)
            
            action, _ = model.predict(obs, deterministic=True)
            
            if action == 1:
                # Logic: Green (even) -> Yellow (odd)
                current = traci.trafficlight.getPhase(tls_id)
                next_p = (current + 1) % 8
                traci.trafficlight.setPhase(tls_id, next_p)
        
        traci.simulationStep()
        step += 1
        
    traci.close()
    
    # Parse Metrics
    try:
        tree = ET.parse(trip_file)
        root = tree.getroot()
        trips = root.findall('tripinfo')
        if not trips: return 0, 0
        
        avg_wait = sum([float(t.get('waitingTime')) for t in trips]) / len(trips)
        return len(trips), avg_wait
    except Exception as e:
        print(f"Error parsing {trip_file}: {e}")
        return 0, 0

def main():
    print("Running Quick Evaluation (1000 steps)...")
    
    # 1. Fixed Time
    print("1. Running Fixed Time...")
    n_fixed, wait_fixed = run_simulation(model=None, steps=1000, label="Fixed")
    
    # 2. RL Agent
    print("2. Running RL Agent...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_stage8_asymmetric")
    model = PPO.load(model_path)
    n_rl, wait_rl = run_simulation(model=model, steps=1000, label="RL")
    
    print("\n--- RESULTS ---")
    print(f"Fixed Time: Avg Wait = {wait_fixed:.2f}s (Over {n_fixed} trips)")
    print(f"RL Agent:   Avg Wait = {wait_rl:.2f}s (Over {n_rl} trips)")
    
    if wait_rl < wait_fixed:
        print("WINNER: RL Agent")
    else:
        print("WINNER: Fixed Time (Needs more training)")

if __name__ == "__main__":
    main()
