import traci
import sumolib
import os
import sys

# Path to config
base_path = os.path.dirname(os.path.abspath(__file__))
cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")

def debug_topology():
    print("Starting SUMO to check topology...")
    sumo_binary = sumolib.checkBinary('sumo')
    cmd = [sumo_binary, "-c", cfg_file, "--no-step-log", "true"]
    
    traci.start(cmd)
    
    junctions = ["JN", "JE", "JS", "JW", "J0"]
    
    with open("topology.txt", "w") as f:
        f.write("--- ACTUAL INCOMING LANES (FROM SUMO) ---\n")
        for j_id in junctions:
            f.write(f"\nJunction {j_id}:\n")
            controlled = traci.trafficlight.getControlledLanes(j_id)
            unique_lanes = sorted(list(set(controlled)))
            for lane in unique_lanes:
                edge = traci.lane.getEdgeID(lane)
                f.write(f"  Lane: {lane} (Edge: {edge})\n")
            
    traci.close()
    print("\n--- END TOPOLOGY CHECK ---")

if __name__ == "__main__":
    debug_topology()
