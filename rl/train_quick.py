import os
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_env import SumoIntersectionEnv

def train_quick():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    dummy_route = "dummy"
    
    models_dir = os.path.join(base_path, "models")
    log_dir = os.path.join(base_path, "logs")
    
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    # Use GUI=False for speed
    env = SumoIntersectionEnv(cfg_file, dummy_route, use_gui=False, num_seconds=20000, use_sumocfg=True)
    
    print("Setting up PPO Agent (Quick)...")
    # MlpPolicy is fast
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    print("Starting QUICK training (8000 steps)...")
    model.learn(total_timesteps=8000, reset_num_timesteps=False, tb_log_name="PPO_Quick")
    
    model.save(f"{models_dir}/ppo_marl")
    print("Quick Training complete. Model saved to ppo_marl.")
    env.close()

if __name__ == "__main__":
    train_quick()
