from flask import Flask, render_template, jsonify
import threading
import json
import os
import sys

# Ensure rl and traci modules are in path
sys.path.append(os.path.join(os.path.dirname(__file__), "rl"))
sys.path.append(os.path.join(os.path.dirname(__file__), "traci"))
import sim_runner

app = Flask(__name__)

# Global lock for simulation (only one at a time)
sim_lock = threading.Lock()
current_status = "Idle"

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
