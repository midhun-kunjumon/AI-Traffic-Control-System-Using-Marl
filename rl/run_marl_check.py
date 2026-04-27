import os
import sys
import sim_runner

def main():
    steps = 1000
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marl_check.log")
    
    if os.path.exists(log_file):
        os.remove(log_file)
        
    print(f"Running Stable RL Simulation (Headless) for {steps} steps...")
    print(f"Collision warnings will be logged to: {log_file}")
    
    # Run simulation
    sim_runner.run_stable_rl(
        steps=steps, 
        headless=True, 
        collision_check=True, 
        log_file=log_file
    )
    
    print("Simulation complete.")
    
    # Analyze Log
    if os.path.exists(log_file):
        print("\n--- ANALYSIS ---")
        with open(log_file, 'r') as f:
            content = f.read()
            
        warnings = [line for line in content.splitlines() if "Collision" in line or "teleport" in line or "Warning" in line]
        
        if warnings:
            print(f"Found {len(warnings)} issues:")
            for w in warnings:
                print(w)
        else:
            print("No collisions, teleportations, or warnings found.")
    else:
        print("Log file was not created!")

if __name__ == "__main__":
    main()
