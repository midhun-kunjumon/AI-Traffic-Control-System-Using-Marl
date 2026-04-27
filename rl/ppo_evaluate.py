import os
import sys
import traci
import sumolib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import json
import xml.etree.ElementTree as ET

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "..", "traci"))

# Import system components
try:
    from sim_runner import TrafficAgent, JUNCTION_CONFIG, ALL_JUNCTIONS
    from emergency_manager import EmergencyManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Define fallback if import fails (to keep script standalone-ish if needed, but better to fix path)
    sys.exit(1)

# Configuration
SUMO_BINARY = sumolib.checkBinary('sumo') # Use 'sumo' for faster headless, or 'sumo-gui' for visual
# If user wants visual, they can change this. Defaulting to headless for evaluation speed?
# User said "Evaluation only (inference mode)", usually implies faster. But earlier "sumo-gui" was used.
# Let's use 'sumo' (headless) for evaluation efficiency unless user asked for GUI.
# User constraints: "Use the existing SUMO environment".
# "Run multiple evaluation episodes (e.g., 5–10 runs)".
# I'll use headless 'sumo' for speed.

CONFIG_FILE = os.path.join(current_dir, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
MODEL_PATH = os.path.join(current_dir, "models", "ppo_marl")
RESULTS_DIR = os.path.join(current_dir, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class TrafficEvaluator:
    def __init__(self, model_path, episodes=5, steps_per_episode=1000):
        self.model_path = model_path
        self.episodes = episodes
        self.steps = steps_per_episode
        self.results = []
        
        # Load Model
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = PPO.load(self.model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def evaluate(self):
        print(f"Starting Evaluation: {self.episodes} episodes per mode, {self.steps} steps per episode.")
        modes = ["fixed", "ppo"]
        
        all_metrics = []
        
        for mode in modes:
            print(f"\n--- Evaluating Mode: {mode.upper()} ---")
            for i in range(self.episodes):
                metrics = self.run_episode(mode, i)
                all_metrics.append(metrics)
                print(f"Episode {i+1}: Wait={metrics['avg_waiting_time']:.2f}s, Queue={metrics['avg_queue_length']:.2f}, Throughput={metrics['throughput']}")
                
        # Save to CSV
        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        self.plot_results(df)
        
        # Print Summary
        summary = df.groupby("mode").agg(["mean", "std"])
        print("\n=== FINAL SUMMARY ===")
        print(summary)

    def run_episode(self, mode, episode_idx):
        # Unique tripinfo file for this run
        trip_file = os.path.join(RESULTS_DIR, f"tripinfo_{mode}_{episode_idx}.xml")
        
        # Base Command
        cmd = [SUMO_BINARY, "-c", CONFIG_FILE, 
               "--tripinfo-output", trip_file,
               "--no-step-log", "true", 
               "--waiting-time-memory", "1000",
               "--start", "--quit-on-end"]
               
        traci.start(cmd)
        
        # Setup Agents if PPO
        agents = {}
        if mode == "ppo":
            emg_manager = EmergencyManager(tls_id="J0") # Placeholder
            for j_id in ALL_JUNCTIONS:
                # Use TrafficAgent from sim_runner
                # Note: TrafficAgent expects 'model' to have .predict()
                agents[j_id] = TrafficAgent(j_id, self.model, emg_manager, enable_emergency=False)
        
        # Metric Accumulators
        total_queue_sum = 0
        step_count = 0
        
        # Identify all lanes to monitor (from JUNCTION_CONFIG)
        monitored_lanes = []
        for j_config in JUNCTION_CONFIG.values():
            for dir_lanes in j_config.values():
                monitored_lanes.extend(dir_lanes)
        monitored_lanes = list(set(monitored_lanes))
        
        step = 0
        while step < self.steps:
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
                
            # Agent Action
            if mode == "ppo":
                for agent in agents.values():
                    agent.act(step)
            
            traci.simulationStep()
            
            # Collect Step Metrics
            # Queue Length: Sum of halting vehicles on all controller lanes
            # This is a good proxy for "System Queue"
            current_q = 0
            for lane in monitored_lanes:
                current_q += traci.lane.getLastStepHaltingNumber(lane)
            
            total_queue_sum += current_q
            step_count += 1
            step += 1
            
        traci.close()
        
        # Post-process Tripinfo for Waiting Time & Throughput
        avg_wait = 0.0
        throughput = 0
        
        if os.path.exists(trip_file):
            try:
                tree = ET.parse(trip_file)
                root = tree.getroot()
                trips = root.findall('tripinfo')
                throughput = len(trips)
                if throughput > 0:
                    avg_wait = sum([float(t.get('waitingTime')) for t in trips]) / throughput
            except Exception as e:
                print(f"Warning: Failed to parse tripinfo: {e}")
            
            # Cleanup
            try:
                os.remove(trip_file)
            except: pass
            
        avg_queue = total_queue_sum / step_count if step_count > 0 else 0.0
        
        return {
            "mode": mode,
            "episode": episode_idx + 1,
            "avg_waiting_time": avg_wait,
            "avg_queue_length": avg_queue,
            "throughput": throughput
        }

    def plot_results(self, df):
        metrics = ["avg_waiting_time", "avg_queue_length", "throughput"]
        titles = ["Average Waiting Time (s)", "Average Queue Length (vehs)", "Throughput (veh/episode)"]
        colors = ['#d62728', '#1f77b4'] # Red (Fixed), Blue (PPO) - usually convention
        
        # Check if we have both modes
        modes = df['mode'].unique()
        
        for metric, title in zip(metrics, titles):
            plt.figure(figsize=(10, 6))
            
            # Calculate stats
            means = df.groupby("mode")[metric].mean()
            stds = df.groupby("mode")[metric].std()
            
            # Plot
            x = range(len(modes))
            plt.bar(x, means, yerr=stds, capsize=10, color=colors[:len(modes)], alpha=0.7)
            plt.xticks(x, [m.upper() for m in modes])
            plt.title(f"Performance Comparison: {title}")
            plt.ylabel(title)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Add text labels
            for i, v in enumerate(means):
                plt.text(i, v + (0.05 * v), f"{v:.2f}", ha='center', fontweight='bold')
            
            save_path = os.path.join(RESULTS_DIR, f"{metric}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    # Check if we are in main directory or rl directory
    # Adjust paths if needed?
    # The script uses os.path.abspath(__file__) so it should be fine.
    
    evaluator = TrafficEvaluator(MODEL_PATH, episodes=5, steps_per_episode=1000)
    evaluator.evaluate()
