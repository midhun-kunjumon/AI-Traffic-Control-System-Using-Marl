from sumo_env import SumoIntersectionEnv
import os
import time

def debug():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    
    print("Initializing Env...")
    env = SumoIntersectionEnv(cfg_file, "dummy", use_gui=False, num_seconds=20000, use_sumocfg=True)
    
    obs, info = env.reset()
    print("Env Reset. Starting Loop...")
    
    start_time = time.time()
    for i in range(100):
        action = 0 if i % 10 != 0 else 1 # Switch every 10 steps
        obs, reward, done, truncated, info = env.step(action)
        if i % 10 == 0:
            print(f"Step {i}: Reward={reward} Phase={obs[0]}")
        if done:
            break
            
    print(f"100 steps took {time.time() - start_time:.2f} seconds")
    env.close()

if __name__ == "__main__":
    debug()
