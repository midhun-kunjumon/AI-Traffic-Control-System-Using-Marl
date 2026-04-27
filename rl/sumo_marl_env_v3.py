import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
import numpy as np
import os
import sys
import random
import logging

# --- JUNCTION CONFIGURATION (mirrors sim_runner.py) ---
JUNCTION_CONFIG = {
    "J0": {
        "N": ["JN_to_J0_0", "JN_to_J0_1"],
        "E": ["JE_to_J0_0", "JE_to_J0_1"],
        "S": ["JS_to_J0_0", "JS_to_J0_1"],
        "W": ["JW_to_J0_0", "JW_to_J0_1"]
    },
    "JN": {
        "N": ["N_to_JN_0", "N_to_JN_1"],
        "E": ["JN_E_to_JN_0", "JN_E_to_JN_1"],
        "S": ["J0_to_JN_0", "J0_to_JN_1"],
        "W": ["JN_W_to_JN_0", "JN_W_to_JN_1"]
    },
    "JS": {
        "N": ["J0_to_JS_0", "J0_to_JS_1"],
        "E": ["JS_E_to_JS_0", "JS_E_to_JS_1"],
        "S": ["S_to_JS_0", "S_to_JS_1"],
        "W": ["JS_W_to_JS_0", "JS_W_to_JS_1"]
    },
    "JW": {
        "N": ["JW_N_to_JW_0", "JW_N_to_JW_1"],
        "E": ["J0_to_JW_0", "J0_to_JW_1"],
        "S": ["JW_S_to_JW_0", "JW_S_to_JW_1"],
        "W": ["W_to_JW_0", "W_to_JW_1"]
    },
    "JE": {
        "N": ["JE_N_to_JE_0", "JE_N_to_JE_1"],
        "E": ["E_to_JE_0", "E_to_JE_1"],
        "S": [],
        "W": ["J0_to_JE_0", "J0_to_JE_1"]
    }
}

ALL_JUNCTIONS = ["J0", "JN", "JS", "JW", "JE"]

# Junction weights for biased sampling — busier junctions sampled more
JUNCTION_WEIGHTS = {
    "J0": 0.35,   # Center junction (heaviest traffic)
    "JN": 0.20,   # North (heavy N-S corridor)
    "JS": 0.20,   # South (heavy N-S corridor)
    "JW": 0.125,  # West (lighter traffic)
    "JE": 0.125   # East (lighter, 3-way)
}


class SumoMarlEnvV3(gym.Env):
    """
    MARL V3 Training Environment — Improved over V1 with:
    - 21-dim observation (phase, phase_dur, 4×queue, 4×count, 4×wait, 4×density, 4×speed)
    - Pressure-based reward with delta tracking
    - Observation normalization
    - Weighted junction selection for parameter sharing
    - Collision-safe yellow transitions
    - Detailed reward component logging
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, use_gui=False, num_seconds=20000,
                 decision_interval=5.0, min_green=10.0, yellow_duration=3.0,
                 log_dir=None):
        super(SumoMarlEnvV3, self).__init__()

        self._cfg_file = cfg_file
        self._gui = use_gui
        self._max_steps = num_seconds  # matched to SUMO cfg end=2000
        self._decision_interval = decision_interval
        self._min_green = min_green
        self._yellow_duration = yellow_duration

        self._step = 0
        self._sim_step = 0
        self._tls_id = "J0"
        self._last_switch_step = 0
        self._last_phase = 0
        self._num_phases = 8
        self._prev_total_wait = 0.0
        self._prev_total_queue = 0.0

        # Episode tracking for logging
        self._episode_num = 0
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_switches = 0
        self._episode_rewards_history = []

        # Reward component accumulators for logging
        self._ep_pressure_sum = 0.0
        self._ep_queue_penalty_sum = 0.0
        self._ep_wait_delta_sum = 0.0
        self._ep_throughput_sum = 0.0
        self._ep_switch_penalty_sum = 0.0

        # Action Space: 0 = Keep, 1 = Switch
        self.action_space = spaces.Discrete(2)

        # Observation Space (Size 21):
        # [phase_norm, phase_dur_norm,
        #  Q_N, Q_E, Q_S, Q_W,
        #  C_N, C_E, C_S, C_W,
        #  W_N, W_E, W_S, W_W,
        #  D_N, D_E, D_S, D_W,
        #  S_N, S_E, S_S, S_W]  (but that's 22, let me recount)
        # Actually: phase(1) + phase_dur(1) + queue(4) + count(4) + wait(4) + density(4) + speed(4) = 22
        # Let's keep it at 21 by merging phase into normalized form:
        # [phase_norm, phase_dur_norm, Q(4), C(4), W(4), D(4), S(3)] = 21
        # Actually let's just use 22 for clarity
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(22,), dtype=np.float32)

        # Setup logging
        self._setup_logging(log_dir)

        if self._gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

    def _setup_logging(self, log_dir):
        """Setup detailed reward logging to file."""
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)

        self._reward_log_path = os.path.join(log_dir, "training_reward_details.log")
        self._logger = logging.getLogger("MarlV3Rewards")
        self._logger.setLevel(logging.INFO)
        # Clear existing handlers
        self._logger.handlers = []

        fh = logging.FileHandler(self._reward_log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

        # Write header
        self._logger.info("=" * 120)
        self._logger.info("MARL V3 TRAINING -- REWARD FUNCTION DETAILS LOG")
        self._logger.info("=" * 120)
        self._logger.info(f"{'Episode':>8} | {'Step':>6} | {'Junction':>4} | {'Action':>6} | "
                          f"{'Pressure':>10} | {'MaxQueue':>10} | {'WaitDelta':>10} | "
                          f"{'Throughput':>10} | {'SwitchPen':>10} | {'StepReward':>12} | {'EpReward':>12}")
        self._logger.info("-" * 120)

    def _get_sumo_cmd(self):
        return [
            self._sumo_binary,
            "-c", self._cfg_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--collision.check-junctions", "true",
            "--collision.action", "warn",
            "--start", "--quit-on-end"
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Log episode summary before reset (if not first episode)
        if self._episode_num > 0:
            self._log_episode_summary()

        try:
            traci.close()
        except:
            pass

        traci.start(self._get_sumo_cmd())
        self._step = 0
        self._sim_step = 0
        self._last_switch_step = 0
        self._prev_total_wait = 0.0
        self._prev_total_queue = 0.0

        # Weighted random junction selection
        junctions = list(JUNCTION_WEIGHTS.keys())
        weights = [JUNCTION_WEIGHTS[j] for j in junctions]
        self._tls_id = random.choices(junctions, weights=weights, k=1)[0]
        self._config = JUNCTION_CONFIG[self._tls_id]

        if self._tls_id == "JE":
            self._num_phases = 6
        else:
            self._num_phases = 8

        self._last_phase = traci.trafficlight.getPhase(self._tls_id)

        # Reset episode trackers
        self._episode_num += 1
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_switches = 0
        self._ep_pressure_sum = 0.0
        self._ep_queue_penalty_sum = 0.0
        self._ep_wait_delta_sum = 0.0
        self._ep_throughput_sum = 0.0
        self._ep_switch_penalty_sum = 0.0

        self._logger.info(f"\n{'='*120}")
        self._logger.info(f"EPISODE {self._episode_num} START | Junction: {self._tls_id} | Phases: {self._num_phases}")
        self._logger.info(f"{'='*120}")

        return self._get_observation(), {}

    def step(self, action):
        self._step += 1
        self._episode_steps += 1
        current_phase = traci.trafficlight.getPhase(self._tls_id)
        dt = traci.simulation.getDeltaT()

        time_since_switch = (self._sim_step - self._last_switch_step) * dt

        switched = False
        original_action = action

        # Enforce minimum green time
        if action == 1 and time_since_switch < self._min_green:
            action = 0

        if action == 1:
            # --- COLLISION-SAFE SWITCH: Green -> Yellow -> Next Green ---
            yellow_phase = (current_phase + 1) % self._num_phases
            traci.trafficlight.setPhase(self._tls_id, yellow_phase)

            yellow_steps = max(1, int(self._yellow_duration / dt))
            for _ in range(yellow_steps):
                traci.simulationStep()
                self._sim_step += 1
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break

            next_green = (yellow_phase + 1) % self._num_phases
            traci.trafficlight.setPhase(self._tls_id, next_green)
            self._last_switch_step = self._sim_step
            switched = True
            self._episode_switches += 1

        # Run simulation for decision interval
        interval_steps = max(1, int(self._decision_interval / dt))
        departed_before = traci.simulation.getDepartedNumber()

        for _ in range(interval_steps):
            traci.simulationStep()
            self._sim_step += 1
            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        departed_after = traci.simulation.getDepartedNumber()

        # --- REWARD COMPUTATION (Pressure-Based) ---
        all_lanes = []
        for direction in ["N", "E", "S", "W"]:
            all_lanes.extend(self._config[direction])

        active_lanes = [l for l in all_lanes if l]

        total_waiting = sum([traci.lane.getWaitingTime(l) for l in active_lanes])
        total_queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in active_lanes])
        throughput = departed_after - departed_before

        # Per-direction queues for max-queue penalty
        dir_queues = []
        for direction in ["N", "E", "S", "W"]:
            lanes = self._config[direction]
            if lanes:
                dir_queues.append(sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes]))
            else:
                dir_queues.append(0)
        max_queue = max(dir_queues) if dir_queues else 0

        # Pressure: incoming queue minus throughput
        pressure = total_queue - max(0, throughput)

        # Wait delta (improvement tracking)
        wait_delta = total_waiting - self._prev_total_wait
        self._prev_total_wait = total_waiting
        self._prev_total_queue = total_queue

        # --- Reward Components (Scaled to reasonable range) ---
        r_pressure = -0.01 * pressure
        r_max_queue = -0.05 * max_queue  # Scaled down from -0.5 to prevent domination
        r_wait_delta = -0.001 * max(0, wait_delta)
        r_throughput = 0.5 * max(0, throughput)
        r_switch = -1.0 if switched else 0.0

        reward = r_pressure + r_max_queue + r_wait_delta + r_throughput + r_switch

        # Accumulate for episode summary
        self._episode_reward += reward
        self._ep_pressure_sum += r_pressure
        self._ep_queue_penalty_sum += r_max_queue
        self._ep_wait_delta_sum += r_wait_delta
        self._ep_throughput_sum += r_throughput
        self._ep_switch_penalty_sum += r_switch

        # Log every step with full reward breakdown
        action_str = "SWITCH" if switched else ("KEEP" if original_action == 0 else "HOLD_MIN")
        self._logger.info(
            f"{self._episode_num:>8} | {self._episode_steps:>6} | {self._tls_id:>4} | "
            f"{action_str:>6} | "
            f"{r_pressure:>10.4f} | {r_max_queue:>10.4f} | {r_wait_delta:>10.4f} | "
            f"{r_throughput:>10.4f} | {r_switch:>10.4f} | {reward:>12.4f} | {self._episode_reward:>12.4f}"
        )

        obs = self._get_observation()
        done = traci.simulation.getMinExpectedNumber() <= 0 or self._step >= self._max_steps
        truncated = False

        if done:
            self._log_episode_summary()

        return obs, reward, done, truncated, {
            "reward_pressure": r_pressure,
            "reward_max_queue": r_max_queue,
            "reward_wait_delta": r_wait_delta,
            "reward_throughput": r_throughput,
            "reward_switch": r_switch,
            "total_queue": total_queue,
            "total_waiting": total_waiting,
            "throughput": throughput,
            "junction": self._tls_id,
        }

    def _log_episode_summary(self):
        """Log a detailed episode summary."""
        avg_reward = self._episode_reward / max(1, self._episode_steps)
        self._episode_rewards_history.append(self._episode_reward)

        # Moving average over last 10 episodes
        recent = self._episode_rewards_history[-10:]
        moving_avg = sum(recent) / len(recent)

        # Compute improvement trend
        if len(self._episode_rewards_history) >= 2:
            prev = self._episode_rewards_history[-2]
            improvement = self._episode_reward - prev
            trend = "IMPROVING ^" if improvement > 0 else "DECLINING v" if improvement < 0 else "STABLE --"
        else:
            improvement = 0
            trend = "BASELINE"

        self._logger.info(f"\n{'-'*120}")
        self._logger.info(f"EPISODE {self._episode_num} SUMMARY | Junction: {self._tls_id}")
        self._logger.info(f"{'-'*120}")
        self._logger.info(f"  Total Steps:      {self._episode_steps}")
        self._logger.info(f"  Total Switches:   {self._episode_switches}")
        self._logger.info(f"  TOTAL REWARD:     {self._episode_reward:>12.4f}  (avg/step: {avg_reward:>8.4f})")
        self._logger.info(f"  10-Ep Moving Avg: {moving_avg:>12.4f}")
        self._logger.info(f"  Trend:            {trend} (delta: {improvement:>+.4f})")
        self._logger.info(f"  -- Reward Component Breakdown --")
        self._logger.info(f"    Pressure:       {self._ep_pressure_sum:>12.4f}")
        self._logger.info(f"    Max Queue:      {self._ep_queue_penalty_sum:>12.4f}")
        self._logger.info(f"    Wait Delta:     {self._ep_wait_delta_sum:>12.4f}")
        self._logger.info(f"    Throughput:      {self._ep_throughput_sum:>12.4f}")
        self._logger.info(f"    Switch Penalty: {self._ep_switch_penalty_sum:>12.4f}")
        self._logger.info(f"{'-'*120}\n")

    def _get_observation(self):
        """Build 22-dim normalized observation."""
        phase = traci.trafficlight.getPhase(self._tls_id)
        dt = traci.simulation.getDeltaT()
        phase_duration = (self._sim_step - self._last_switch_step) * dt

        # Normalize phase to [0, 1]
        phase_norm = phase / max(1, self._num_phases - 1)
        # Normalize phase duration (cap at 120s)
        phase_dur_norm = min(phase_duration / 120.0, 1.0)

        queues = []
        counts = []
        waits = []
        densities = []
        speeds = []

        for direction in ["N", "E", "S", "W"]:
            lanes = self._config[direction]
            if not lanes:
                queues.append(0.0)
                counts.append(0.0)
                waits.append(0.0)
                densities.append(0.0)
                speeds.append(0.0)
            else:
                q = sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
                c = sum([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
                w = sum([traci.lane.getWaitingTime(l) for l in lanes])

                # Density: vehicles per meter of lane
                total_length = sum([traci.lane.getLength(l) for l in lanes])
                d = c / max(1.0, total_length) if total_length > 0 else 0.0

                # Mean speed across lanes
                s_vals = [traci.lane.getLastStepMeanSpeed(l) for l in lanes]
                s = sum(s_vals) / len(s_vals) if s_vals else 0.0

                queues.append(float(q))
                counts.append(float(c))
                waits.append(float(w))
                densities.append(float(d))
                speeds.append(float(s))

        # Normalize all features to [0, 1]
        norm_queues = [min(q / 30.0, 1.0) for q in queues]
        norm_counts = [min(c / 40.0, 1.0) for c in counts]
        norm_waits = [min(w / 500.0, 1.0) for w in waits]
        norm_densities = [min(d / 0.15, 1.0) for d in densities]
        norm_speeds = [min(s / 15.0, 1.0) for s in speeds]

        obs = np.array(
            [phase_norm, phase_dur_norm] +
            norm_queues + norm_counts + norm_waits +
            norm_densities + norm_speeds,
            dtype=np.float32
        )
        return obs

    def close(self):
        # Final logging
        if self._episode_steps > 0:
            self._log_episode_summary()

        self._logger.info(f"\n{'='*120}")
        self._logger.info(f"TRAINING COMPLETE -- Total Episodes: {self._episode_num}")
        if self._episode_rewards_history:
            first_10 = self._episode_rewards_history[:10]
            last_10 = self._episode_rewards_history[-10:]
            self._logger.info(f"  First 10 Avg Reward: {sum(first_10)/len(first_10):>12.4f}")
            self._logger.info(f"  Last  10 Avg Reward: {sum(last_10)/len(last_10):>12.4f}")
            overall_improvement = (sum(last_10)/len(last_10)) - (sum(first_10)/len(first_10))
            self._logger.info(f"  Overall Improvement: {overall_improvement:>+12.4f}")
        self._logger.info(f"{'='*120}")

        try:
            traci.close()
        except:
            pass
