import sys
import os

# Add RL directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

import sim_runner

print("Starting Headless Debug Run...")
try:
    n, wait = sim_runner.run_stable_rl(steps=2000, enable_emergency=True, headless=True)
    print(f"Run Complete. N={n}, Wait={wait}")
except Exception as e:
    print(f"Run Failed: {e}")
