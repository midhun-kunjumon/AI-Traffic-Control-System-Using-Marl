"""
AI Traffic Control System — Production Pipeline
=================================================
ESP32 Cameras → YOLOv8 Detection → RL Decision → Serial → ESP32 Traffic Lights

Data Flow:
  1. Capture ONE frame from each of 4 ESP32 camera streams
  2. Run YOLOv8 inference → count vehicles + detect emergency
  3. If emergency → IMMEDIATE priority (skip RL)
  4. Otherwise   → RL agent decides priority lane
  5. Send decision to ESP32 via Serial (COM7)
  6. ESP32 executes signal (GREEN for selected lane)
  7. After signal duration → all YELLOW → next cycle

Usage:
    python traffic_system.py
    python traffic_system.py --port COM7 --conf 0.3

Author: AI Traffic Control Project
"""

import cv2
import time
import sys
import os
import json
import logging
import numpy as np
import urllib.request
from datetime import datetime
from collections import deque

import serial
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# -- YOLOv8 Model --
MODEL_PATH = r"yolov8\best1.pt"
CONFIDENCE_THRESHOLD = 0.3

# -- Camera Streams (ESP32 — corrected mapping) --
LANE_STREAMS = {
    "N": "http://192.168.1.3",
    "E": "http://192.168.1.6",
    "S": "http://192.168.1.7",
    "W": "http://192.168.1.5",
}
LANE_ORDER = ["N", "E", "S", "W"]

# -- Serial Communication --
SERIAL_PORT = "COM7"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 2  # seconds

# -- Signal Timing --
DEFAULT_GREEN_TIME = 15       # seconds (base green time)
MIN_GREEN_TIME = 8            # seconds minimum
MAX_GREEN_TIME = 45           # seconds maximum
YELLOW_DURATION = 3           # seconds
EMERGENCY_GREEN_TIME = 20     # seconds for emergency priority

# -- RL Parameters --
RL_HISTORY_SIZE = 20          # past cycles to remember
DEMAND_WEIGHT_QUEUE = 2.0     # weight for queue-based demand
DEMAND_WEIGHT_WAIT = 1.5      # weight for waiting-time demand

# -- Logging --
LOG_DIR = "traffic_logs"
DETECTION_DIR = "detection_results"

# ═══════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DETECTION_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger("TrafficSystem")


# ═══════════════════════════════════════════════════════════════════
#  FRAME CAPTURE
# ═══════════════════════════════════════════════════════════════════

def grab_frame(url, lane_name, timeout=10):
    """
    Grab a single JPEG frame from an ESP32 camera.
    Tries /capture endpoint first, then falls back to MJPEG stream.
    Returns numpy image or None on failure.
    """
    # Method 1: HTTP snapshot endpoints
    for endpoint in ["/capture", "/jpg", ""]:
        try:
            full_url = url.rstrip("/") + endpoint
            req = urllib.request.Request(full_url)
            resp = urllib.request.urlopen(req, timeout=timeout)
            img_bytes = bytearray(resp.read())
            img_array = np.asarray(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        except Exception:
            continue

    # Method 2: OpenCV VideoCapture (MJPEG stream)
    for endpoint in ["/stream", ":81/stream", ""]:
        try:
            cap = cv2.VideoCapture(url.rstrip("/") + endpoint)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame
        except Exception:
            continue

    logger.warning(f"Could not grab frame from Lane {lane_name} ({url})")
    return None


# ═══════════════════════════════════════════════════════════════════
#  YOLOV8 DETECTION MODULE
# ═══════════════════════════════════════════════════════════════════

class VehicleDetector:
    """YOLOv8 vehicle detection wrapper."""

    def __init__(self, model_path=MODEL_PATH, conf=CONFIDENCE_THRESHOLD):
        logger.info(f"Loading YOLOv8 model: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            sys.exit(1)

        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names  # {0: 'cars', 1: 'emergency'}
        logger.info(f"Model loaded. Classes: {self.class_names}")

    def detect(self, frame):
        """
        Run inference on a single frame.

        Returns:
            dict: {
                "cars": int,
                "emergency": int,
                "total": int,
                "has_emergency": bool,
                "annotated_frame": np.ndarray
            }
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        result = results[0]

        cars = 0
        emergency = 0

        # Support both standard bounding boxes and oriented bounding boxes (OBB)
        boxes_obj = result.boxes if result.boxes is not None else result.obb

        # Debug: log raw detection info
        num_boxes = len(boxes_obj) if boxes_obj is not None else 0
        logger.debug(f"  [DEBUG] Raw boxes/obb: {num_boxes}, type: {type(boxes_obj)}")

        if boxes_obj is not None and len(boxes_obj) > 0:
            for i, box in enumerate(boxes_obj):
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = self.class_names[cls_id].lower()
                logger.debug(f"  [DEBUG] Box {i}: class={class_name} "
                             f"(id={cls_id}), conf={conf_val:.2f}")
                if class_name == "emergency":
                    emergency += 1
                else:
                    cars += 1

        return {
            "cars": cars,
            "emergency": emergency,
            "total": cars + emergency,
            "has_emergency": emergency > 0,
            "annotated_frame": result.plot(),
        }


# ═══════════════════════════════════════════════════════════════════
#  REINFORCEMENT LEARNING DECISION MODULE
# ═══════════════════════════════════════════════════════════════════

class RLDecisionModule:
    """
    RL-based traffic signal decision engine.

    Uses a smart rule-based policy approximating RL behavior:
      - Prioritizes lanes with highest demand (vehicles + waiting time)
      - Computes proportional green times
      - Applies fairness constraints (starvation prevention)

    Designed to be easily replaceable with a trained PPO/DQN model.
    """

    def __init__(self):
        self.cycle_count = 0
        self.history = deque(maxlen=RL_HISTORY_SIZE)
        self.last_green_lane = None
        # Track how many cycles since each lane got green (starvation prevention)
        self.cycles_since_green = {lane: 0 for lane in LANE_ORDER}
        logger.info("RL Decision Module initialized (rule-based policy)")

    def decide(self, vehicle_counts, emergency_lanes=None):
        """
        Main decision function.

        Args:
            vehicle_counts: dict {"N": int, "E": int, "S": int, "W": int}
            emergency_lanes: list of lane names with emergency vehicles

        Returns:
            dict: {
                "priority_lane": str,         # "N", "E", "S", or "W"
                "green_times": dict,           # {"N": sec, "E": sec, ...}
                "reason": str,                 # human-readable decision reason
                "is_emergency": bool,
            }
        """
        self.cycle_count += 1

        # ── EMERGENCY OVERRIDE ─────────────────────────────────────
        if emergency_lanes:
            # Pick the first emergency lane (or the one with most vehicles)
            if len(emergency_lanes) == 1:
                priority = emergency_lanes[0]
            else:
                # Multiple emergencies: pick lane with most total vehicles
                priority = max(emergency_lanes,
                               key=lambda l: vehicle_counts.get(l, 0))

            green_times = {lane: 0 for lane in LANE_ORDER}
            green_times[priority] = EMERGENCY_GREEN_TIME

            self._update_state(priority, vehicle_counts)

            return {
                "priority_lane": priority,
                "green_times": green_times,
                "reason": f"🚨 EMERGENCY in Lane {priority}",
                "is_emergency": True,
            }

        # ── NORMAL RL DECISION ─────────────────────────────────────

        # Calculate demand score per lane
        demands = {}
        for lane in LANE_ORDER:
            count = vehicle_counts.get(lane, 0)
            starvation_bonus = min(self.cycles_since_green[lane] * 2, 10)
            demand = (DEMAND_WEIGHT_QUEUE * count) + starvation_bonus
            demands[lane] = demand

        total_demand = sum(demands.values())

        # Pick priority lane (highest demand)
        if total_demand == 0:
            # No vehicles anywhere — round-robin
            if self.last_green_lane is None:
                priority = "N"
            else:
                idx = LANE_ORDER.index(self.last_green_lane)
                priority = LANE_ORDER[(idx + 1) % 4]
            reason = "No vehicles detected — round-robin"
        else:
            priority = max(demands, key=demands.get)
            reason = (f"Highest demand: Lane {priority} "
                      f"(score={demands[priority]:.1f}, "
                      f"vehicles={vehicle_counts.get(priority, 0)})")

        # Compute proportional green times
        green_times = {}
        if total_demand > 0:
            available_time = DEFAULT_GREEN_TIME * 4 - (YELLOW_DURATION * 4)
            for lane in LANE_ORDER:
                proportion = demands[lane] / total_demand
                gt = proportion * available_time
                gt = max(gt, MIN_GREEN_TIME)
                gt = min(gt, MAX_GREEN_TIME)
                green_times[lane] = round(gt, 1)
        else:
            green_times = {lane: DEFAULT_GREEN_TIME for lane in LANE_ORDER}

        self._update_state(priority, vehicle_counts)

        return {
            "priority_lane": priority,
            "green_times": green_times,
            "reason": reason,
            "is_emergency": False,
        }

    def _update_state(self, green_lane, vehicle_counts):
        """Update internal tracking state."""
        self.last_green_lane = green_lane
        for lane in LANE_ORDER:
            if lane == green_lane:
                self.cycles_since_green[lane] = 0
            else:
                self.cycles_since_green[lane] += 1

        self.history.append({
            "cycle": self.cycle_count,
            "counts": dict(vehicle_counts),
            "green": green_lane,
            "time": datetime.now().isoformat(),
        })


# ═══════════════════════════════════════════════════════════════════
#  SERIAL COMMUNICATION MODULE
# ═══════════════════════════════════════════════════════════════════

class SerialController:
    """
    Manages serial communication with ESP32 traffic light controller.

    Protocol:
        Send "X\\n" where X is the priority lane: N, E, S, or W.
        ESP32 will:
          1. Turn all RED
          2. Turn selected lane GREEN
          3. Run for duration
          4. Turn GREEN→YELLOW→RED
          5. Wait for next command

    Alternative: Send "N:10,E:15,S:8,W:12\\n" for per-lane timings.
    """

    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False

    def connect(self):
        """Open serial connection to ESP32."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=SERIAL_TIMEOUT,
            )
            # Prevent ESP32 from being held in reset/bootloader mode by PySerial defaults
            self.ser.setDTR(False)
            self.ser.setRTS(False)
            
            time.sleep(2)  # Wait for ESP32 to reset after serial open
            self.connected = True
            logger.info(f"Serial connected: {self.port} @ {self.baud} baud")

            # Flush any startup messages from ESP32
            while self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    logger.info(f"  ESP32 startup: {line}")

            return True
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            self.connected = False
            return False

    def send_priority(self, lane):
        """
        Send priority lane command to ESP32.
        Format: "N\\n" or "E\\n" etc.
        """
        if not self.connected:
            logger.warning("Serial not connected — skipping send")
            return False

        try:
            command = f"{lane}\n"
            self.ser.write(command.encode("utf-8"))
            self.ser.flush()
            logger.info(f"  → Sent to ESP32: {lane}")

            # Read acknowledgment
            time.sleep(0.1)
            while self.ser.in_waiting:
                response = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if response:
                    logger.info(f"  ← ESP32: {response}")

            return True
        except serial.SerialException as e:
            logger.error(f"Serial write error: {e}")
            self.connected = False
            return False

    def send_timings(self, green_times):
        """
        Send per-lane green timings to ESP32.
        Format: "N:10,E:15,S:8,W:12\\n"
        """
        if not self.connected:
            logger.warning("Serial not connected — skipping send")
            return False

        try:
            parts = [f"{lane}:{int(green_times[lane])}" for lane in LANE_ORDER]
            command = ",".join(parts) + "\n"
            self.ser.write(command.encode("utf-8"))
            self.ser.flush()
            logger.info(f"  → Sent timings: {command.strip()}")

            time.sleep(0.1)
            while self.ser.in_waiting:
                response = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if response:
                    logger.info(f"  ← ESP32: {response}")

            return True
        except serial.SerialException as e:
            logger.error(f"Serial write error: {e}")
            self.connected = False
            return False

    def wait_for_ready(self, timeout=60):
        """Wait for ESP32 to send READY signal after completing a cycle."""
        if not self.connected:
            return True  # Don't block if disconnected

        start = time.time()
        while time.time() - start < timeout:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        logger.info(f"  ← ESP32: {line}")
                    if "READY" in line.upper():
                        return True
                time.sleep(0.1)
            except serial.SerialException:
                return True
        logger.warning("Timeout waiting for ESP32 READY — proceeding anyway")
        return False

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Serial disconnected")
        self.connected = False


# ═══════════════════════════════════════════════════════════════════
#  MAIN TRAFFIC CONTROL PIPELINE
# ═══════════════════════════════════════════════════════════════════

class TrafficControlPipeline:
    """
    Main pipeline orchestrator.

    Each cycle:
      1. Capture frames from all 4 cameras
      2. YOLOv8 detection on each frame
      3. Emergency check → priority override
      4. RL decision → best lane
      5. Send to ESP32 via serial
      6. Wait for cycle completion
      7. Repeat
    """

    def __init__(self, serial_port=SERIAL_PORT):
        logger.info("=" * 70)
        logger.info("  AI TRAFFIC CONTROL SYSTEM — INITIALIZING")
        logger.info("=" * 70)

        # Initialize modules
        self.detector = VehicleDetector()
        self.rl_agent = RLDecisionModule()
        self.serial_ctrl = SerialController(port=serial_port)

        # Connect to ESP32
        logger.info(f"\nConnecting to ESP32 on {serial_port}...")
        if not self.serial_ctrl.connect():
            logger.warning("⚠️  ESP32 not connected — running in SIMULATION mode")
            logger.warning("    (decisions will be logged but not sent)")

        self.cycle_count = 0
        self.running = False

    def capture_all_frames(self):
        """Capture one frame from each lane camera."""
        frames = {}
        for lane, url in LANE_STREAMS.items():
            frame = grab_frame(url, lane)
            frames[lane] = frame
        return frames

    def detect_all_lanes(self, frames):
        """
        Run YOLOv8 on all frames.

        Returns:
            detections: dict {lane: detection_result}
            vehicle_counts: dict {lane: total_count}
            emergency_lanes: list of lanes with emergency vehicles
        """
        detections = {}
        vehicle_counts = {}
        emergency_lanes = []

        for lane in LANE_ORDER:
            frame = frames.get(lane)
            if frame is None:
                detections[lane] = {
                    "cars": 0, "emergency": 0, "total": 0,
                    "has_emergency": False, "annotated_frame": None,
                }
                vehicle_counts[lane] = 0
                continue

            det = self.detector.detect(frame)
            detections[lane] = det
            vehicle_counts[lane] = det["total"]

            if det["has_emergency"]:
                emergency_lanes.append(lane)

        return detections, vehicle_counts, emergency_lanes

    def save_detections(self, detections, cycle_num):
        """Save annotated detection images for this cycle."""
        for lane, det in detections.items():
            if det["annotated_frame"] is not None:
                filename = f"cycle{cycle_num}_{lane}.jpg"
                filepath = os.path.join(DETECTION_DIR, filename)
                cv2.imwrite(filepath, det["annotated_frame"])

    def display_detections(self, detections, cycle_num):
        """Show detection images in OpenCV windows (optional)."""
        try:
            for lane, det in detections.items():
                if det["annotated_frame"] is not None:
                    window_name = f"Lane {lane}"
                    display = cv2.resize(det["annotated_frame"], (480, 360))
                    cv2.imshow(window_name, display)
            cv2.waitKey(2000)
        except Exception:
            pass  # Headless environment

    def run_cycle(self):
        """Execute one complete traffic control cycle."""
        self.cycle_count += 1
        cycle_start = time.time()

        logger.info("")
        logger.info("━" * 70)
        logger.info(f"  CYCLE {self.cycle_count}")
        logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("━" * 70)

        # ── Step 1: Capture frames ───────────────────────────────
        logger.info("\n  📷 STEP 1: Capturing frames from cameras...")
        frames = self.capture_all_frames()
        captured = sum(1 for f in frames.values() if f is not None)
        logger.info(f"     Captured: {captured}/4 frames")

        if captured == 0:
            logger.error("     ❌ No frames captured! Skipping cycle.")
            return None

        # ── Step 2: YOLOv8 Detection ─────────────────────────────
        logger.info("\n  🔍 STEP 2: Running YOLOv8 detection...")
        detect_start = time.time()
        detections, vehicle_counts, emergency_lanes = self.detect_all_lanes(frames)
        detect_time = time.time() - detect_start

        # Print detection results
        for lane in LANE_ORDER:
            det = detections[lane]
            if frames[lane] is not None:
                status = f"🚗 {det['cars']} cars, 🚑 {det['emergency']} emergency  [Total: {det['total']}]"
                if det["has_emergency"]:
                    status += "  🚨 EMERGENCY!"
            else:
                status = "❌ No frame"
            logger.info(f"     Lane {lane}: {status}")

        # Print compact vehicle count
        count_str = ", ".join(f"{l}-{vehicle_counts[l]}" for l in LANE_ORDER)
        logger.info(f"\n     📊 Vehicle Counts: {count_str}")
        logger.info(f"     ⏱️  Detection time: {detect_time:.2f}s")

        # Save detection images
        self.save_detections(detections, self.cycle_count)

        # Show images (optional)
        self.display_detections(detections, self.cycle_count)

        # ── Step 3: Emergency Check ──────────────────────────────
        if emergency_lanes:
            logger.info(f"\n  🚨 STEP 3: EMERGENCY DETECTED in lane(s): {', '.join(emergency_lanes)}")
        else:
            logger.info("\n  ✅ STEP 3: No emergency vehicles — proceeding to RL")

        # ── Step 4: RL Decision ──────────────────────────────────
        logger.info("\n  🧠 STEP 4: RL Decision Engine...")
        decision = self.rl_agent.decide(vehicle_counts, emergency_lanes)

        logger.info(f"     Decision: {decision['reason']}")
        logger.info(f"     Priority Lane: {decision['priority_lane']}")
        logger.info(f"     Green Times: { {l: f'{t}s' for l, t in decision['green_times'].items()} }")

        # ── Step 5: Send to ESP32 ────────────────────────────────
        logger.info(f"\n  📡 STEP 5: Sending to ESP32 → Lane {decision['priority_lane']} GREEN")
        self.serial_ctrl.send_priority(decision['priority_lane'])

        # ── Step 6: Wait for signal cycle ────────────────────────
        green_time = decision['green_times'].get(decision['priority_lane'], DEFAULT_GREEN_TIME)
        total_wait = green_time + YELLOW_DURATION
        logger.info(f"\n  ⏳ STEP 6: Waiting for signal cycle ({green_time}s green + {YELLOW_DURATION}s yellow)...")

        # Wait for ESP32 to finish or timeout
        self.serial_ctrl.wait_for_ready(timeout=total_wait + 5)

        # ── Cycle Summary ────────────────────────────────────────
        cycle_time = time.time() - cycle_start
        logger.info(f"\n  ┌{'─' * 50}┐")
        logger.info(f"  │  CYCLE {self.cycle_count} COMPLETE — {cycle_time:.1f}s total")
        logger.info(f"  │  Counts: {count_str}")
        logger.info(f"  │  Decision: Lane {decision['priority_lane']} GREEN"
                     f" ({'EMERGENCY' if decision['is_emergency'] else 'RL'})")
        logger.info(f"  └{'─' * 50}┘")

        return decision

    def run(self):
        """Main loop — run continuously until interrupted."""
        self.running = True

        logger.info("\n" + "=" * 70)
        logger.info("  🚦  SYSTEM ONLINE — Starting traffic control loop")
        logger.info("      Press Ctrl+C to stop")
        logger.info("=" * 70)

        try:
            while self.running:
                self.run_cycle()
        except KeyboardInterrupt:
            logger.info("\n\n⛔ System interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        logger.info("\n" + "=" * 70)
        logger.info("  SHUTTING DOWN")
        logger.info(f"  Total cycles completed: {self.cycle_count}")
        logger.info(f"  Detection images: ./{DETECTION_DIR}/")
        logger.info(f"  Log file: {log_filename}")
        logger.info("=" * 70)
        self.serial_ctrl.disconnect()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Traffic Control System — YOLOv8 + RL + ESP32"
    )
    parser.add_argument(
        "--port", type=str, default=SERIAL_PORT,
        help=f"Serial port for ESP32 (default: {SERIAL_PORT})"
    )
    parser.add_argument(
        "--baud", type=int, default=SERIAL_BAUD,
        help=f"Serial baud rate (default: {SERIAL_BAUD})"
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"YOLOv8 confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV image display windows"
    )
    args = parser.parse_args()

    # Apply overrides
    if args.conf != CONFIDENCE_THRESHOLD:
        CONFIDENCE_THRESHOLD = args.conf

    # Start the pipeline
    pipeline = TrafficControlPipeline(serial_port=args.port)

    if args.no_display:
        pipeline.display_detections = lambda *a, **k: None

    pipeline.run()
