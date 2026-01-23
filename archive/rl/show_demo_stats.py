
import xml.etree.ElementTree as ET
import os

def parse(f):
    if not os.path.exists(f): return "N/A", "N/A"
    try:
        root = ET.parse(f).getroot()
        trips = root.findall('tripinfo')
        if not trips: return 0, 0.0
        avg = sum([float(t.get('waitingTime')) for t in trips]) / len(trips)
        return len(trips), avg
    except:
        return "Error", "Error"

# Demo Files
f_fixed = "tripinfo_demo_fixed.xml"
f_rl = "tripinfo_demo_rl.xml"

n_f, w_f = parse(f_fixed)
n_r, w_r = parse(f_rl)

print("--- DEMO RESULTS ---")
print(f"Fixed Time: {n_f} vehicles, Avg Wait: {w_f:.2f}s")
print(f"RL Agent:   {n_r} vehicles, Avg Wait: {w_r:.2f}s")
