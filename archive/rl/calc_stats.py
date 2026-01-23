import xml.etree.ElementTree as ET
import os

def get_avg(file):
    root = ET.parse(file).getroot()
    infos = root.findall('tripinfo')
    if not infos: return 0.0
    return sum(float(x.get('waitingTime')) for x in infos) / len(infos)

print(f"Fixed: {get_avg('rl/tripinfo_fixed.xml'):.2f}")
print(f"RL V2: {get_avg('rl/tripinfo_rl_v2.xml'):.2f}")
