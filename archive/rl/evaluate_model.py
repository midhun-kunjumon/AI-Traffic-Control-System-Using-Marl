import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import traci
import sumolib
import numpy as np
import xml.etree.ElementTree as ET

from sumo_env import SumoIntersectionEnv

def run_fixed_time(cfg_file, output_file):
    print("Running Fixed Time Simulation...")
    sumo_binary = sumolib.checkBinary('sumo')
    cmd = [
        sumo_binary,
        "-c", cfg_file,
        "--tripinfo-output", output_file,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "-e", "500" # End time
    ]
    traci.start(cmd)
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()
    print(f"Fixed Time finished in {step} steps.")

def run_rl_agent(model_path, cfg_file, output_file):
    print("Running RL Agent Simulation...")
    
    # Initialize Environment with Stage 8 Config
    # We use use_sumocfg=True matching training
    env = SumoIntersectionEnv(cfg_file, "dummy", use_gui=False, num_seconds=500, use_sumocfg=True)
    
    # Inject tripinfo output into the environment's command
    # This must be done before reset() starts the simulation
    env.sumo_cmd.append("--tripinfo-output")
    env.sumo_cmd.append(output_file)
    
    # Load Model
    model = PPO.load(model_path)
    
    obs,info = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
    env.close()
    print(f"RL Agent finished.")

def parse_tripinfo(xml_file):
    if not os.path.exists(xml_file):
         return {"count": 0, "avg_duration":0, "avg_waiting_time":0, "avg_time_loss":0}
         
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    durations = []
    waiting_times = []
    time_losses = []
    
    for trip in root.findall('tripinfo'):
        durations.append(float(trip.get('duration')))
        waiting_times.append(float(trip.get('waitingTime')))
        time_losses.append(float(trip.get('timeLoss')))
        
    return {
        "count": len(durations),
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "avg_waiting_time": sum(waiting_times) / len(waiting_times) if waiting_times else 0,
        "avg_time_loss": sum(time_losses) / len(time_losses) if time_losses else 0
    }

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    
    fixed_output = os.path.join(base_path, "tripinfo_fixed.xml")
    rl_output = os.path.join(base_path, "tripinfo_rl_stage8.xml")
    model_path = os.path.join(base_path, "models", "ppo_stage8_asymmetric")

    # 1. Run Fixed Time
    if os.path.exists(fixed_output): os.remove(fixed_output)
    run_fixed_time(cfg_file, fixed_output)
    
    # 2. Run RL Agent
    if os.path.exists(rl_output): os.remove(rl_output)
    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip! Train properly first.")
        return

    run_rl_agent(model_path, cfg_file, rl_output)
    
    # 3. Compare
    print("\n--- RESULTS STAGE 8 (Asymmetric Traffic) ---")
    stats_fixed = parse_tripinfo(fixed_output)
    stats_rl = parse_tripinfo(rl_output)
    
    print(f"{'Metric':<20} | {'Fixed Time':<15} | {'RL Agent':<15}")
    print("-" * 56)
    print(f"{'Total Cars':<20} | {stats_fixed['count']:<15} | {stats_rl['count']:<15}")
    print(f"{'Avg Waiting Time':<20} | {stats_fixed['avg_waiting_time']:<15.2f} | {stats_rl['avg_waiting_time']:<15.2f}")
    print(f"{'Avg Time Loss':<20} | {stats_fixed['avg_time_loss']:<15.2f} | {stats_rl['avg_time_loss']:<15.2f}")
    
    # Determine Winner
    if stats_rl['avg_waiting_time'] < stats_fixed['avg_waiting_time']:
        print("\n🏆 RL Agent Wins! (Lower waiting time)")
    else:
        print("\n🤖 Fixed Time Wins! (RL needs more training)")

if __name__ == "__main__":
    main()
