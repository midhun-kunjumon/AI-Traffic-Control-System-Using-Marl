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

# Import TrafficAgent and configuration from sim_runner
# We need to make sure we can import sim_runner. 
# Since we are in the same directory, we can import directly.
try:
    from sim_runner import TrafficAgent, ALL_JUNCTIONS, RESULTS_FILE, parse_stats, save_result
except ImportError:
    # If running from root, adjust path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from sim_runner import TrafficAgent, ALL_JUNCTIONS, RESULTS_FILE, parse_stats, save_result

def run_check(steps=1000, enable_emergency=False):
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    trip_file_name = "tripinfo_check.xml"
    trip_file = os.path.join(base_path, trip_file_name)
    model_path = os.path.join(base_path, "models", "ppo_marl")
    log_file = os.path.join(base_path, "simulation_check.log")
    
    if os.path.exists(trip_file): os.remove(trip_file)
    if os.path.exists(log_file): os.remove(log_file)
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    emg_manager = EmergencyManager(tls_id="J0") # Placeholder
    
    # Use sumolib to find sumo binary (headless)
    sumo_binary = sumolib.checkBinary('sumo') 
    
    cmd = [
        sumo_binary, "-c", cfg_file, "--tripinfo-output", trip_file,
        "--no-step-log", "true", "--waiting-time-memory", "1000",
        "-e", str(steps), "--start", "--quit-on-end",
        "--collision.check-junctions", "true",
        "--collision.action", "warn",
        "--log", log_file
    ]
    
    print(f"Running simulation with command: {' '.join(cmd)}")
    traci.start(cmd)
    
    # Initialize Agents
    agents = {}
    for j_id in ALL_JUNCTIONS:
        agents[j_id] = TrafficAgent(j_id, model, emg_manager, enable_emergency=enable_emergency)
        
    step = 0
    collisions = 0
    
    print("Simulation started...")
    while step < steps:
        if traci.simulation.getMinExpectedNumber() <= 0: break

        # Check for collisions/teleports via TraCI directly if possible, or rely on log
        colliding = traci.simulation.getCollidingVehiclesNumber()
        if colliding > 0:
            collisions += colliding
            # print(f"Step {step}: {colliding} collisions detected.")
            
        for agent in agents.values():
            agent.act(step)

        traci.simulationStep()
        step += 1
            
    traci.close()
    print("Simulation finished.")
    
    print(f"Checking log file: {log_file}")
    
    # Parse log file for warnings
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
            
        print("\n--- SIMULATION REPORT ---")
        found_issues = False
        for line in log_content.splitlines():
            if "Collision" in line or "teleport" in line or "Warning" in line:
                print(line)
                found_issues = True
        
        if not found_issues:
            print("No collisions, teleportations, or warnings detected in the log.")
    else:
        print("Log file not found!")

if __name__ == "__main__":
    run_check()
