"""
Real-Time Traffic Controller
=============================
Connects ESP32 camera streams → YOLOv8 detection → PPO MARL model → signal timings.

This is the main integration module that:
  1. Grabs live frames from 4 ESP32 cameras (North, East, South, West)
  2. Runs YOLOv8 vehicle detection on each frame
  3. Builds a 22-dim observation vector matching the trained PPO model format
  4. Feeds the observation into the MARL PPO model for keep/switch decisions
  5. Computes green-time allocations for each direction

The Flask app (app.py) imports and runs this controller via:
  - /run_live_camera  → starts the controller
  - /get_live_status  → returns current detection + signal state
  - /stop_live_camera → stops the controller
"""

import os
import sys
import time
import numpy as np
from collections import deque

# Add project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "yolov8"))
sys.path.append(os.path.join(BASE_DIR, "rl"))

from stable_baselines3 import PPO
from yolo_detector import VehicleDetector
from camera_stream import CameraManager, CAMERA_URLS


# =============================================================================
# Configuration
# =============================================================================

# Path to the trained MARL PPO model
PPO_MODEL_PATH = os.path.join(BASE_DIR, "rl", "models", "ppo_marl_v3")

# Total traffic signal cycle time (seconds)
DEFAULT_CYCLE_TIME = 120.0

# Minimum green time per direction (seconds)
MIN_GREEN_TIME = 10.0

# Yellow phase duration (seconds)
YELLOW_DURATION = 5.0

# Assumed lane length in meters (for density normalization — match training env)
ASSUMED_LANE_LENGTH = 200.0

# Number of recent frames to keep for waiting time estimation
WAITING_TIME_WINDOW = 30  # frames

# Directions (must match camera configuration)
DIRECTIONS = ["North", "East", "South", "West"]

# PPO model observation normalization caps (must match sumo_marl_env_v3.py)
NORM_QUEUE_MAX = 30.0
NORM_COUNT_MAX = 40.0
NORM_WAIT_MAX = 500.0
NORM_DENSITY_MAX = 0.15
NORM_SPEED_MAX = 15.0  # m/s


class RealTimeTrafficController:
    """
    Real-time traffic signal controller using ESP32 cameras + YOLOv8 + PPO.
    
    Flow per step:
      cameras → YOLOv8 → observation vector → PPO prediction → signal timing
    """

    def __init__(self, cycle_time=DEFAULT_CYCLE_TIME, camera_urls=None,
                 camera_mode="snapshot"):
        """
        Args:
            cycle_time: Total cycle time in seconds for green allocation.
            camera_urls: Optional dict overriding default camera URLs.
            camera_mode: "snapshot" or "mjpeg" for ESP32 camera capture.
        """
        self.cycle_time = cycle_time
        self.step_count = 0

        # --- Load PPO Model ---
        print(f"[Controller] Loading PPO model from: {PPO_MODEL_PATH}")
        self.model = PPO.load(PPO_MODEL_PATH)
        print(f"[Controller] PPO model loaded successfully.")

        # --- Initialize YOLOv8 Detector ---
        self.detector = VehicleDetector()

        # --- Initialize Camera Manager ---
        urls = camera_urls or CAMERA_URLS
        self.camera_manager = CameraManager(camera_urls=urls, mode=camera_mode)
        self.camera_manager.start_all()

        # --- Traffic State ---
        self.current_phase = 0          # 0=North green, 2=East green, 4=South green, 6=West green
        self.phase_start_time = time.time()
        self.num_phases = 8             # Even=green, Odd=yellow (matching training env)

        # Per-direction metrics (latest)
        self.lane_metrics = {d: {
            "vehicle_count": 0,
            "queue_length": 0,
            "moving_count": 0,
            "waiting_time": 0.0,
            "density": 0.0,
            "speed": 0.0,
        } for d in DIRECTIONS}

        # Waiting time estimation: track queue history per direction
        # Waiting time ≈ sum of seconds each vehicle has been halting
        self._queue_history = {d: deque(maxlen=WAITING_TIME_WINDOW) for d in DIRECTIONS}
        self._last_step_time = time.time()

        # Signal timing output
        self.green_times = {d: cycle_time / 4.0 for d in DIRECTIONS}
        self.current_green_direction = "North"
        self.signal_state = {d: "RED" for d in DIRECTIONS}
        self.signal_state["North"] = "GREEN"

        # PPO action history
        self.last_action = 0
        self.action_history = deque(maxlen=50)

        print(f"[Controller] Initialized. Cycle time: {cycle_time}s")
        print(f"[Controller] Camera URLs:")
        for d, url in urls.items():
            print(f"  {d:6s}: {url}")

    def step(self):
        """
        Execute one control cycle:
          1. Capture frames from all cameras
          2. Run YOLOv8 detection
          3. Build observation
          4. Run PPO inference
          5. Update signal timings

        Returns:
            dict with detection results and signal timing decisions
        """
        self.step_count += 1
        now = time.time()
        dt = now - self._last_step_time
        self._last_step_time = now

        # --- 1. Capture frames from all 4 cameras ---
        frames = self.camera_manager.get_all_frames()

        # --- 2. Run YOLOv8 detection on each frame ---
        detections = {}
        for direction in DIRECTIONS:
            frame = frames.get(direction)
            if frame is not None:
                det = self.detector.detect(frame, camera_id=direction)
            else:
                det = self.detector._empty_result()
            detections[direction] = det

        # --- 3. Update lane metrics from detections ---
        for direction in DIRECTIONS:
            det = detections[direction]
            queue = det["queue_length"]
            count = det["vehicle_count"]
            speed_px = det["avg_speed_estimate"]

            # Convert pixel speed to approximate m/s
            # Rough: 1 pixel ≈ 0.05m at typical ESP32-CAM resolution/distance
            PIXEL_TO_METER = 0.05
            speed_ms = speed_px * PIXEL_TO_METER

            # Density: vehicles per lane length
            density = count / ASSUMED_LANE_LENGTH

            # Accumulate waiting time: queue_length × dt
            self._queue_history[direction].append(queue)
            # Estimated waiting time = average queue over window × window duration
            avg_queue = np.mean(list(self._queue_history[direction])) if self._queue_history[direction] else 0
            waiting_time = avg_queue * len(self._queue_history[direction]) * dt

            self.lane_metrics[direction] = {
                "vehicle_count": count,
                "queue_length": queue,
                "moving_count": det["moving_count"],
                "waiting_time": waiting_time,
                "density": density,
                "speed": speed_ms,
            }

        # --- 4. Build 22-dim observation vector (matching ppo_marl_v3 training) ---
        obs = self._build_observation()

        # --- 5. Run PPO model inference ---
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        self.last_action = action
        self.action_history.append(action)

        # --- 6. Process action: keep or switch ---
        phase_duration = now - self.phase_start_time

        if action == 1 and phase_duration >= MIN_GREEN_TIME:
            # Switch to next direction
            self._switch_phase()

        # --- 7. Compute proportional green time allocation ---
        self._compute_green_times()

        result = {
            "step": self.step_count,
            "detections": {d: {
                "vehicle_count": detections[d]["vehicle_count"],
                "queue_length": detections[d]["queue_length"],
                "moving_count": detections[d]["moving_count"],
            } for d in DIRECTIONS},
            "lane_metrics": self.lane_metrics,
            "signal": {
                "current_green": self.current_green_direction,
                "phase": self.current_phase,
                "phase_duration": round(phase_duration, 1),
                "action": "SWITCH" if action == 1 else "KEEP",
                "signal_state": dict(self.signal_state),
            },
            "green_times": dict(self.green_times),
            "cameras": self.camera_manager.get_status(),
        }

        # Log summary to console
        if self.step_count % 5 == 1:
            self._log_step(result)

        return result

    def _build_observation(self):
        """
        Build the 22-dimensional observation vector matching the PPO training env.
        
        Format (from sumo_marl_env_v3.py):
          [phase_norm, phase_dur_norm,
           Q_N, Q_E, Q_S, Q_W,          (queue lengths, normalized)
           C_N, C_E, C_S, C_W,          (vehicle counts, normalized)
           W_N, W_E, W_S, W_W,          (waiting times, normalized)
           D_N, D_E, D_S, D_W,          (densities, normalized)
           S_N, S_E, S_S, S_W]          (speeds, normalized)
        """
        # Phase normalization
        phase_norm = self.current_phase / max(1, self.num_phases - 1)

        # Phase duration normalization (cap at 120s)
        phase_duration = time.time() - self.phase_start_time
        phase_dur_norm = min(phase_duration / 120.0, 1.0)

        queues = []
        counts = []
        waits = []
        densities = []
        speeds = []

        for direction in DIRECTIONS:
            m = self.lane_metrics[direction]
            queues.append(min(m["queue_length"] / NORM_QUEUE_MAX, 1.0))
            counts.append(min(m["vehicle_count"] / NORM_COUNT_MAX, 1.0))
            waits.append(min(m["waiting_time"] / NORM_WAIT_MAX, 1.0))
            densities.append(min(m["density"] / NORM_DENSITY_MAX, 1.0))
            speeds.append(min(m["speed"] / NORM_SPEED_MAX, 1.0))

        obs = np.array(
            [phase_norm, phase_dur_norm] +
            queues + counts + waits +
            densities + speeds,
            dtype=np.float32
        )

        return obs

    def _switch_phase(self):
        """Switch to the next green direction (density-based smart switch)."""
        # Find direction with highest queue to prioritize
        max_queue = -1
        best_direction = None

        for d in DIRECTIONS:
            if d == self.current_green_direction:
                continue  # Skip current direction
            q = self.lane_metrics[d]["queue_length"]
            if q > max_queue:
                max_queue = q
                best_direction = d

        if best_direction is None:
            # Cycle to next direction
            idx = DIRECTIONS.index(self.current_green_direction)
            best_direction = DIRECTIONS[(idx + 1) % len(DIRECTIONS)]

        # Update signal state
        self.signal_state[self.current_green_direction] = "RED"
        self.current_green_direction = best_direction
        self.signal_state[best_direction] = "GREEN"

        # Update phase index (0=North, 2=East, 4=South, 6=West)
        direction_to_phase = {"North": 0, "East": 2, "South": 4, "West": 6}
        self.current_phase = direction_to_phase[best_direction]
        self.phase_start_time = time.time()

        print(f"[Controller] SWITCH → {best_direction} GREEN (queue={max_queue})")

    def _compute_green_times(self):
        """
        Compute proportional green-time split based on vehicle density/queue.
        
        Total cycle time is divided proportionally to each direction's
        vehicle demand (queue + count), with a minimum guaranteed time.
        """
        demands = {}
        total_demand = 0

        for d in DIRECTIONS:
            m = self.lane_metrics[d]
            # Demand = queue weight (heavier) + count weight
            demand = 2.0 * m["queue_length"] + m["vehicle_count"]
            demands[d] = max(demand, 0.1)  # Minimum non-zero
            total_demand += demands[d]

        # Available time after yellow phases (4 directions × YELLOW_DURATION)
        available_time = self.cycle_time - (4 * YELLOW_DURATION)
        available_time = max(available_time, 4 * MIN_GREEN_TIME)

        for d in DIRECTIONS:
            proportion = demands[d] / total_demand
            green = proportion * available_time
            green = max(green, MIN_GREEN_TIME)
            self.green_times[d] = round(green, 1)

    def get_status(self):
        """
        Get the current controller status for the Flask API.
        
        Returns:
            dict with all detection data, signal state, and green times.
        """
        phase_duration = time.time() - self.phase_start_time

        return {
            "running": True,
            "step_count": self.step_count,
            "current_green": self.current_green_direction,
            "phase_duration": round(phase_duration, 1),
            "signal_state": dict(self.signal_state),
            "green_times": dict(self.green_times),
            "lane_metrics": {
                d: {
                    "vehicle_count": self.lane_metrics[d]["vehicle_count"],
                    "queue_length": self.lane_metrics[d]["queue_length"],
                    "waiting_time": round(self.lane_metrics[d]["waiting_time"], 1),
                    "density": round(self.lane_metrics[d]["density"], 4),
                    "speed": round(self.lane_metrics[d]["speed"], 2),
                } for d in DIRECTIONS
            },
            "cameras": self.camera_manager.get_status(),
            "last_action": "SWITCH" if self.last_action == 1 else "KEEP",
        }

    def _log_step(self, result):
        """Print a summary of the current step to console."""
        det = result["detections"]
        sig = result["signal"]
        gt = result["green_times"]

        print(f"\n{'='*70}")
        print(f"  Step {result['step']} | Green: {sig['current_green']} "
              f"({sig['phase_duration']}s) | Action: {sig['action']}")
        print(f"{'='*70}")
        print(f"  {'Direction':>9} | {'Vehicles':>8} | {'Queue':>5} | "
              f"{'Moving':>6} | {'Green Time':>10}")
        print(f"  {'-'*9}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*10}")
        for d in DIRECTIONS:
            signal = "🟢" if self.signal_state[d] == "GREEN" else "🔴"
            print(f"  {signal} {d:>7} | {det[d]['vehicle_count']:>8} | "
                  f"{det[d]['queue_length']:>5} | {det[d]['moving_count']:>6} | "
                  f"{gt[d]:>8.1f}s")
        print(f"{'='*70}")

    def stop(self):
        """Stop the controller and release resources."""
        self.camera_manager.stop_all()
        self.detector.reset_tracking()
        print("[Controller] Stopped.")


# =============================================================================
# Standalone Run (for testing without Flask)
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-Time Traffic Controller")
    parser.add_argument("--cycle-time", type=float, default=120.0,
                        help="Total cycle time in seconds (default: 120)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Detection interval in seconds (default: 5)")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with synthetic frames")
    args = parser.parse_args()

    if args.test:
        print("\n" + "=" * 60)
        print("  TEST MODE — Using synthetic frames")
        print("=" * 60)

        # Create a synthetic test controller that generates fake frames
        import cv2

        class TestCameraManager:
            """Generates random synthetic frames for testing."""
            def start_all(self): pass
            def stop_all(self): pass

            def get_all_frames(self):
                frames = {}
                for d in DIRECTIONS:
                    # Create a blank frame (640x480 BGR)
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Add some random "vehicle-like" rectangles
                    n_vehicles = np.random.randint(0, 15)
                    for _ in range(n_vehicles):
                        x = np.random.randint(50, 590)
                        y = np.random.randint(50, 430)
                        w = np.random.randint(20, 60)
                        h = np.random.randint(15, 40)
                        color = (
                            np.random.randint(100, 255),
                            np.random.randint(100, 255),
                            np.random.randint(100, 255),
                        )
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
                    frames[d] = frame
                return frames

            def get_status(self):
                return {d: {"online": True, "last_frame_age": 0.1, "url": "test://"}
                        for d in DIRECTIONS}

        # Initialize controller
        controller = RealTimeTrafficController(cycle_time=args.cycle_time)
        controller.camera_manager.stop_all()
        controller.camera_manager = TestCameraManager()

        print(f"\nRunning test for 20 steps (interval={args.interval}s)...\n")

        try:
            for i in range(20):
                result = controller.step()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopping test...")
        finally:
            print("\nTest complete.")

    else:
        # Normal mode: connect to real ESP32 cameras
        print("\n" + "=" * 60)
        print("  LIVE MODE — Connecting to ESP32 cameras")
        print("=" * 60)
        print("  Make sure your ESP32 camera URLs are configured in")
        print("  yolov8/camera_stream.py before running!\n")

        controller = RealTimeTrafficController(cycle_time=args.cycle_time)

        try:
            while True:
                result = controller.step()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopping controller...")
        finally:
            controller.stop()
