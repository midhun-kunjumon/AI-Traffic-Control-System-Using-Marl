import os
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Import our custom environment
# Since it's in the same folder, we can import directly or register it
# For simplicity, we import class directly
from sumo_env import SumoIntersectionEnv

def train():
    # Paths
    # We use the Stage 8 Asymmetric map
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Pass the config file as net_file argument when use_sumocfg=True
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    # Route file is unused if sumocfg matches, but Env expects an arg? 
    # Actually Env assumes net_file is config if use_sumocfg=True. We pass None or dummy for route.
    dummy_route = "dummy"
    
    models_dir = os.path.join(base_path, "models")
    log_dir = os.path.join(base_path, "logs")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize Environment
    # Stage 8: Use sumocfg to ensure 2-Lane LHT + Traffic Light settings are loaded
    env = SumoIntersectionEnv(cfg_file, dummy_route, use_gui=False, num_seconds=20000, use_sumocfg=True)
    
    # RL Agent
    print("Setting up PPO Agent for Stage 8 Asymmetric...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    print("Starting training...")
    # Train for 50k timesteps for decent convergence
    TIMESTEPS = 50000
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_MARL")
    
    # Save the model
    model.save(f"{models_dir}/ppo_marl")
    print("Training complete. Model saved.")
    
    env.close()

if __name__ == "__main__":
    train()
