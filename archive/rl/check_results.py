import xml.etree.ElementTree as ET
import os

def parse_stats(trip_file):
    if not os.path.exists(trip_file):
        return 0, 0.0
    try:
        root = ET.parse(trip_file).getroot()
        trips = root.findall('tripinfo')
        if not trips: return 0, 0.0
        avg_wait = sum([float(t.get('waitingTime')) for t in trips]) / len(trips)
        return len(trips), avg_wait
    except Exception as e:
        print(f"Error parsing {trip_file}: {e}")
        return 0, 0.0

def main():
    output = []
    output.append("--- Simulation Results ---")
    
    # Fixed Time
    n_fixed, wait_fixed = parse_stats("rl/tripinfo_demo_fixed.xml")
    output.append(f"[FIXED TIME] Avg Waiting Time: {wait_fixed:.2f} s (over {n_fixed} vehicles)")
    
    # RL Agent
    n_rl, wait_rl = parse_stats("rl/tripinfo_demo_rl.xml")
    output.append(f"[RL AGENT]   Avg Waiting Time: {wait_rl:.2f} s (over {n_rl} vehicles)")
    
    if n_fixed > 0:
        improvement = ((wait_fixed - wait_rl) / wait_fixed) * 100
        output.append(f"\nImprovement: {improvement:.1f}% reduction in waiting time.")
        
    with open("rl/results.txt", "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    main()
