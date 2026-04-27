import os
import sys
import io
import time
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Import our improved environment
from sumo_marl_env_v3 import SumoMarlEnvV3


class DetailedRewardLoggingCallback(BaseCallback):
    """
    Custom callback that logs detailed reward metrics to both
    TensorBoard and a human-readable text file.
    Shows reward improvement over training.
    """

    def __init__(self, log_path, verbose=1):
        super(DetailedRewardLoggingCallback, self).__init__(verbose)
        self.log_path = log_path
        self.training_log_path = os.path.join(log_path, "training_progress.log")
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        self.start_time = None

        # Per-step component tracking
        self.step_rewards = {
            "pressure": [],
            "max_queue": [],
            "wait_delta": [],
            "throughput": [],
            "switch": []
        }

    def _on_training_start(self):
        self.start_time = time.time()

        # Setup text logger
        self._txt_logger = logging.getLogger("TrainingProgress")
        self._txt_logger.setLevel(logging.INFO)
        self._txt_logger.handlers = []

        fh = logging.FileHandler(self.training_log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        self._txt_logger.addHandler(fh)

        # Console handler with safe encoding
        console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        ch = logging.StreamHandler(stream=console_stream)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self._txt_logger.addHandler(ch)

        self._txt_logger.info("=" * 130)
        self._txt_logger.info("MARL V3 TRAINING PROGRESS LOG")
        self._txt_logger.info("Started: %s", time.strftime('%Y-%m-%d %H:%M:%S'))
        self._txt_logger.info("Total Timesteps Target: %d", self.model._total_timesteps)
        self._txt_logger.info("=" * 130)
        self._txt_logger.info("")
        self._txt_logger.info(
            "%10s | %7s | %12s | %12s | %12s | %6s | %10s | %10s | %10s | %10s | %10s | %8s | %12s",
            "Timestep", "Episode", "EpReward", "Mean100", "Best",
            "EpLen", "Pressure", "Queue", "WaitDelta", "Throughput", "Switch", "Time", "Status"
        )
        self._txt_logger.info("-" * 130)

    def _on_step(self):
        # Check for episode completion from infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_count += 1

                # Compute stats
                mean_100 = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                is_best = mean_100 > self.best_mean_reward
                if is_best and len(self.episode_rewards) >= 10:
                    self.best_mean_reward = mean_100

                elapsed = time.time() - self.start_time
                elapsed_str = "%dm%ds" % (int(elapsed // 60), int(elapsed % 60))

                # Determine improvement status
                if len(self.episode_rewards) < 5:
                    status = "WARMUP"
                elif is_best and len(self.episode_rewards) >= 10:
                    status = "NEW BEST *"
                elif len(self.episode_rewards) >= 20:
                    recent_10 = np.mean(self.episode_rewards[-10:])
                    prev_10 = np.mean(self.episode_rewards[-20:-10])
                    if recent_10 > prev_10 * 1.05:
                        status = "IMPROVING ^"
                    elif recent_10 < prev_10 * 0.95:
                        status = "DECLINING v"
                    else:
                        status = "STABLE --"
                else:
                    status = "LEARNING"

                # Get reward components from info
                r_pressure = info.get("reward_pressure", 0)
                r_queue = info.get("reward_max_queue", 0)
                r_wait = info.get("reward_wait_delta", 0)
                r_throughput = info.get("reward_throughput", 0)
                r_switch = info.get("reward_switch", 0)

                # Log to TensorBoard
                self.logger.record("reward/episode_reward", ep_reward)
                self.logger.record("reward/mean_100", mean_100)
                self.logger.record("reward/best_mean", self.best_mean_reward)
                self.logger.record("reward/component_pressure", r_pressure)
                self.logger.record("reward/component_queue", r_queue)
                self.logger.record("reward/component_wait_delta", r_wait)
                self.logger.record("reward/component_throughput", r_throughput)
                self.logger.record("reward/component_switch", r_switch)
                self.logger.record("traffic/total_queue", info.get("total_queue", 0))
                self.logger.record("traffic/total_waiting", info.get("total_waiting", 0))
                self.logger.record("traffic/throughput", info.get("throughput", 0))
                self.logger.record("episode/length", ep_length)
                self.logger.record("episode/count", self.episode_count)

                # Log to text file
                self._txt_logger.info(
                    "%10d | %7d | %12.2f | %12.2f | %12.2f | %6d | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %8s | %12s",
                    self.num_timesteps, self.episode_count, ep_reward,
                    mean_100, self.best_mean_reward, ep_length,
                    r_pressure, r_queue, r_wait,
                    r_throughput, r_switch, elapsed_str, status
                )

                # Periodic summary every 25 episodes
                if self.episode_count % 25 == 0 and self.episode_count > 0:
                    self._log_progress_summary()

        return True

    def _log_progress_summary(self):
        """Log a progress summary block."""
        elapsed = (time.time() - self.start_time) / 60
        progress = self.num_timesteps / self.model._total_timesteps * 100

        recent_25 = self.episode_rewards[-25:]
        first_25 = self.episode_rewards[:25] if len(self.episode_rewards) >= 25 else self.episode_rewards[:len(self.episode_rewards)]

        self._txt_logger.info("")
        self._txt_logger.info("  +============================================================================+")
        self._txt_logger.info("  |  PROGRESS CHECKPOINT -- Episode %5d | Timestep %8d (%.1f%%)  |",
                              self.episode_count, self.num_timesteps, progress)
        self._txt_logger.info("  +============================================================================+")
        self._txt_logger.info("  |  Elapsed Time:         %8.1f min                                     |", elapsed)
        self._txt_logger.info("  |  Mean Reward (last 25): %10.2f                                    |", np.mean(recent_25))
        self._txt_logger.info("  |  Mean Reward (first 25):%10.2f                                    |", np.mean(first_25))
        improvement = np.mean(recent_25) - np.mean(first_25)
        marker = "[+] IMPROVED" if improvement > 0 else "[-] REGRESSED"
        self._txt_logger.info("  |  Improvement:          %+10.2f  (%s)                     |", improvement, marker)
        self._txt_logger.info("  |  Best Mean (100-ep):   %10.2f                                    |", self.best_mean_reward)
        self._txt_logger.info("  +============================================================================+")
        self._txt_logger.info("")

    def _on_training_end(self):
        elapsed = (time.time() - self.start_time) / 60
        self._txt_logger.info("")
        self._txt_logger.info("=" * 130)
        self._txt_logger.info("TRAINING COMPLETE")
        self._txt_logger.info("=" * 130)
        self._txt_logger.info("  Total Time:     %.1f minutes", elapsed)
        self._txt_logger.info("  Total Episodes: %d", self.episode_count)
        self._txt_logger.info("  Total Steps:    %d", self.num_timesteps)

        if len(self.episode_rewards) >= 10:
            first_10 = np.mean(self.episode_rewards[:10])
            last_10 = np.mean(self.episode_rewards[-10:])
            self._txt_logger.info("  First 10 Episodes Avg Reward: %12.4f", first_10)
            self._txt_logger.info("  Last  10 Episodes Avg Reward: %12.4f", last_10)
            self._txt_logger.info("  Overall Improvement:          %+12.4f", last_10 - first_10)

            if last_10 > first_10:
                pct = ((last_10 - first_10) / abs(first_10) * 100) if first_10 != 0 else 0
                self._txt_logger.info("  * RESULT: Agent IMPROVED by %.1f%%", pct)
            else:
                self._txt_logger.info("  * RESULT: Agent did not improve. Consider more training or tuning.")

        self._txt_logger.info("  Best Mean Reward (100-ep window): %.4f", self.best_mean_reward)
        self._txt_logger.info("=" * 130)


def train():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")

    models_dir = os.path.join(base_path, "models")
    log_dir = os.path.join(base_path, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize Improved Environment
    env = SumoMarlEnvV3(
        cfg_file=cfg_file,
        use_gui=False,
        num_seconds=2000,     # Match SUMO cfg end time for shorter episodes
        decision_interval=5.0,
        min_green=10.0,
        yellow_duration=3.0,
        log_dir=log_dir
    )

    # PPO with tuned hyperparameters and larger network
    print("=" * 80)
    print("MARL V3 TRAINING -- PPO with Enhanced Observation & Pressure-Based Reward")
    print("=" * 80)
    print("  Observation Space: %s" % str(env.observation_space.shape))
    print("  Action Space:      %s" % str(env.action_space))
    print("  Network:           [256, 256]")
    print("  Learning Rate:     3e-4")
    print("  Batch Size:        128")
    print("  Timesteps:         50,000")
    print("=" * 80)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=log_dir
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="ppo_marl_v3_checkpoint"
    )

    reward_log_cb = DetailedRewardLoggingCallback(log_path=log_dir)

    # Train
    TIMESTEPS = 50000
    print("\nStarting training for %d timesteps..." % TIMESTEPS)
    print("Reward log: %s" % os.path.join(log_dir, 'training_reward_details.log'))
    print("Progress log: %s" % os.path.join(log_dir, 'training_progress.log'))
    print("Random junction selection per episode for MARL parameter sharing.\n")

    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=True,
        tb_log_name="PPO_MARL_V3",
        callback=[checkpoint_cb, reward_log_cb]
    )

    # Save final model
    final_path = os.path.join(models_dir, "ppo_marl_v3")
    model.save(final_path)
    print("\nTraining complete. Model saved to %s.zip" % final_path)
    print("Check logs at:")
    print("  - %s  (per-step reward breakdown)" % os.path.join(log_dir, 'training_reward_details.log'))
    print("  - %s  (episode progress + improvement)" % os.path.join(log_dir, 'training_progress.log'))

    env.close()


if __name__ == "__main__":
    train()
