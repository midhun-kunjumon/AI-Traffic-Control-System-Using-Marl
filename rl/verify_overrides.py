import sim_runner
import traci
import time

def verify_overrides():
    print("Testing Anti-Starvation and Emergency Overrides...")
    steps = 1000
    # Enable Emergency
    sim_runner.run_stable_rl(steps=steps, enable_emergency=True)
    
    print("\nVerification Complete.")
    print("Check logs above for 'Teleporting' warnings. If none/few, Anti-Starvation is working.")
    print("Observe if Emergency vehicles moved fluently.")

if __name__ == "__main__":
    verify_overrides()
