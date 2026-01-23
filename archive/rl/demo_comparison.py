
import os
import sys
import traci
import sumolib
import numpy as np
from stable_baselines3 import PPO
import xml.etree.ElementTree as ET

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

def run_gui_fixed(steps=1000):
    print("\n" + "="*50)
    print(f"PHASE 1: FIXED TIME CONTROLLER ({steps} steps)")
    print("running SUMO-GUI... (Observation Mode)")
    print("Notice: Traffic on East-West (Left/Right) is light, but green time is wasted.")
    print("="*50 + "\n")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file = os.path.join(base_path, "tripinfo_demo_fixed.xml")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    
    sumo_binary = sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary,
        "-c", cfg_file,
        "--tripinfo-output", trip_file,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "-e", str(steps),
        "--start",
        "--quit-on-end"
    ]
    
    traci.start(cmd)
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        if step >= steps:
            break
            
    traci.close()
    
    n, wait = parse_stats(trip_file)
    print(f"\n[FIXED TIME RESULT] Avg Waiting Time: {wait:.2f} seconds (over {n} vehicles)")

def run_gui_rl(steps=1000):
    print("\n" + "="*50)
    print(f"PHASE 2: STABLE RL AGENT ({steps} steps)")
    print("running SUMO-GUI... (AI Mode)")
    print("Notice: East-West green time is skipped if empty, BUT lights hold min 10s.")
    print("="*50 + "\n")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file = os.path.join(base_path, "tripinfo_demo_rl.xml")
    model_path = os.path.join(base_path, "models", "ppo_stable")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    
    if not os.path.exists(model_path + ".zip"):
         print(f"Model {model_path} not found! Run training first.")
         return

    model = PPO.load(model_path)
    
    sumo_binary = sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary,
        "-c", cfg_file,
        "--tripinfo-output", trip_file,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "-e", str(steps),
        "--start",
        "--quit-on-end"
    ]
    
    traci.start(cmd)
    
    tls_id = "J0"
    step = 0
    last_phase = -1
    phase_start_step = 0
    
    while traci.simulation.getMinExpectedNumber() > 0:
        # RL Logic
        phase = traci.trafficlight.getPhase(tls_id)
        
        # Track Phase Change
        if phase != last_phase:
            last_phase = phase
            phase_start_step = step
            
        # Aggregate lanes (matching Env logic)
        lanes = [
            ("N_to_J0_0", "N_to_J0_1"),
            ("E_to_J0_0", "E_to_J0_1"),
            ("S_to_J0_0", "S_to_J0_1"),
            ("W_to_J0_0", "W_to_J0_1")
        ]
        queues = []
        counts = []
        for l0, l1 in lanes:
            queues.append(traci.lane.getLastStepHaltingNumber(l0) + traci.lane.getLastStepHaltingNumber(l1))
            counts.append(traci.lane.getLastStepVehicleNumber(l0) + traci.lane.getLastStepVehicleNumber(l1))
            
        obs = np.array([phase] + queues + counts, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        
        # Stability Logic: Min Green 10s
        # Stability Logic: Min Green 5s (User Requested)
        # Use traci.simulation.getDeltaT() for accuracy, assume default if needed
        dt = traci.simulation.getDeltaT()
        time_since_change = (step - phase_start_step) * dt
        min_green = 5.0
        
        if action == 1:
             if time_since_change < min_green:
                  # Force Wait (Min Green Constraint)
                  pass
             else:
                 # === SMART SWITCH LOGIC ===
                 # 1. Switch to Yellow
                 current = traci.trafficlight.getPhase(tls_id)
                 yellow_phase = (current + 1) % 8
                 traci.trafficlight.setPhase(tls_id, yellow_phase)
                 
                 # Simulate Yellow Duration (3s)
                 yellow_steps = int(3.0 / dt)
                 for _ in range(yellow_steps):
                     traci.simulationStep()
                     step += 1
                     if traci.simulation.getMinExpectedNumber() <= 0: break
                 
                 # 2. Choose Next Green based on Max Queue (Greedy)
                 # Get queues for current state
                 lane_queues = []
                 for l0, l1 in lanes:
                     # Sum halting vehicles on both lanes of the arm
                     q = traci.lane.getLastStepHaltingNumber(l0) + traci.lane.getLastStepHaltingNumber(l1)
                     lane_queues.append(q)
                 
                 # Current lane index (0=N, 1=E, 2=S, 3=W)
                 current_lane_idx = int(current / 2) 
                 
                 # Mask current lane (don't switch back to self immediately if we decided to leave)
                 # Although if we left, it means RL wanted to leave.
                 lane_queues[current_lane_idx] = -1 
                 
                 # Find best candidate
                 best_idx = np.argmax(lane_queues)
                 
                 # If valid candidate has cars, switch to it. 
                 # If all others are empty (max <= 0), default to next sequential to keep moving?
                 # Or stay? But we already switched to Yellow. We MUST go to a Green now.
                 if lane_queues[best_idx] <= 0:
                     best_idx = (current_lane_idx + 1) % 4
                     
                 next_green_phase = best_idx * 2
                 traci.trafficlight.setPhase(tls_id, next_green_phase)
                 
                 # Update State Trackers
                 # We just started a new Green phase
                 last_phase = next_green_phase
                 phase_start_step = step 
                 
                 # Check loops
                 if step >= steps: break
        
        traci.simulationStep()
        step += 1
        if step >= steps:
            break
            
    traci.close()
    
    n, wait = parse_stats(trip_file)
    print(f"\n[STABLE RL RESULT]  Avg Waiting Time: {wait:.2f} seconds (over {n} vehicles)")
    
def main():
    # User asked for simulation. Let's do 1000 steps each.
    run_gui_fixed(1000)
    input("\nPress Enter to start RL Agent simulation...")
    run_gui_rl(1000)
    print("\nDemo Complete.")

if __name__ == "__main__":
    main()
