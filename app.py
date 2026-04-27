from flask import Flask, render_template, jsonify, request
import threading
import json
import os
import sys
import time

# Ensure rl, traci, and yolov8 modules are in path
sys.path.append(os.path.join(os.path.dirname(__file__), "rl"))
sys.path.append(os.path.join(os.path.dirname(__file__), "traci"))
sys.path.append(os.path.join(os.path.dirname(__file__), "yolov8"))
import sim_runner

app = Flask(__name__)

# Global lock for simulation (only one at a time)
sim_lock = threading.Lock()
current_status = "Idle"

# Live camera controller (global reference for status queries)
live_controller = None
live_controller_lock = threading.Lock()
live_running = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_simulation/<mode>')
def run_simulation(mode):
    global current_status
    if sim_lock.locked():
        return jsonify({"status": "error", "message": "Simulation already running!"})
    
    thread = threading.Thread(target=run_sim_thread, args=(mode,))
    thread.start()
    return jsonify({"status": "success", "message": f"Started {mode} simulation."})

@app.route('/get_results')
def get_results():
    results_file = os.path.join("rl", "sim_results.json")
    data = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
        except: pass
    
    return jsonify({
        "status": current_status,
        "results": data
    })

# ============================================================================
# LIVE CAMERA MODE — ESP32 + YOLOv8 + PPO Traffic Controller
# ============================================================================

@app.route('/run_live_camera')
def run_live_camera():
    """
    Start the live camera traffic controller.
    Optional query params:
        ?cycle_time=120  (total cycle time in seconds)
        ?interval=5      (detection interval in seconds)
    """
    global live_running, current_status

    if live_running:
        return jsonify({"status": "error", "message": "Live camera controller already running!"})
    if sim_lock.locked():
        return jsonify({"status": "error", "message": "A simulation is already running!"})

    cycle_time = request.args.get('cycle_time', 120.0, type=float)
    interval = request.args.get('interval', 5.0, type=float)

    thread = threading.Thread(
        target=run_live_camera_thread,
        args=(cycle_time, interval),
        daemon=True,
    )
    thread.start()
    return jsonify({
        "status": "success",
        "message": f"Live camera controller started (cycle={cycle_time}s, interval={interval}s).",
    })

@app.route('/stop_live_camera')
def stop_live_camera():
    """Stop the live camera controller."""
    global live_running
    live_running = False
    return jsonify({"status": "success", "message": "Live camera controller stopping..."})

@app.route('/get_live_status')
def get_live_status():
    """
    Get real-time traffic status from the live camera controller.
    Returns per-lane vehicle counts, queue lengths, green times, and RL decisions.
    """
    global live_controller, live_running

    if not live_running or live_controller is None:
        return jsonify({
            "status": "not_running",
            "message": "Live camera controller is not running. Start it via /run_live_camera",
        })

    try:
        status = live_controller.get_status()
        return jsonify({"status": "running", "data": status})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


def run_live_camera_thread(cycle_time, interval):
    """Background thread for the live camera traffic controller."""
    global live_controller, live_running, current_status

    from traffic_controller import RealTimeTrafficController

    with live_controller_lock:
        live_running = True
        current_status = "Live Camera Mode"
        try:
            live_controller = RealTimeTrafficController(cycle_time=cycle_time)
            while live_running:
                result = live_controller.step()
                time.sleep(interval)
        except Exception as e:
            current_status = f"Live Camera Error: {str(e)}"
            print(f"Live Camera Error: {e}")
        finally:
            live_running = False
            current_status = "Idle"
            live_controller = None

# ============================================================================

def run_sim_thread(mode):
    global current_status
    with sim_lock:
        current_status = f"Running {mode}..."
        try:
            if mode == "fixed":
                sim_runner.run_fixed()
            elif mode == "stable_rl":
                sim_runner.run_stable_rl()
            elif mode == "emergency_rl":
                sim_runner.run_emergency_rl()
            current_status = "Completed"
        except Exception as e:
            current_status = f"Error: {str(e)}"
            print(f"Simulation Error: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
