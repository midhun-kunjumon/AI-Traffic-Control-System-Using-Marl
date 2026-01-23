
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

f_fixed = "tripinfo_fixed.xml"
f_rl = "tripinfo_rl.xml"

n_f, w_f = parse(f_fixed)
n_r, w_r = parse(f_rl)

print(f"Fixed: {n_f} trips, {w_f:.2f}s avg wait")
print(f"RL:    {n_r} trips, {w_r:.2f}s avg wait")
