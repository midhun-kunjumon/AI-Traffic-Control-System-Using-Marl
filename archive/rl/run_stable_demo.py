
import os
import sys
import traci
import sumolib
import numpy as np
from stable_baselines3 import PPO

def run_stable_demo(steps=2000):
    print("\n" + "="*50)
    print("STABILITY VERIFICATION (Min Green = 10s)")
    print("loading ppo_stable...")
    print("="*50 + "\n")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(base_path, "..", "sumo", "stage8_asymmetric", "intersection.sumocfg")
    model_path = os.path.join(base_path, "models", "ppo_stable")
    
    if not os.path.exists(model_path + ".zip"):
        print("Model ppo_stable not found yet!")
        return

    model = PPO.load(model_path)
    
    sumo_binary = sumolib.checkBinary('sumo-gui')
    cmd = [
        sumo_binary,
        "-c", cfg_file,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "-e", str(steps),
        "--start",
        "--quit-on-end"
    ]
    
    traci.start(cmd)
    tls_id = "J0"
    
    # Tracking Phase Durations
    last_phase = -1
    phase_start_step = 0
    step = 0
    
    while traci.simulation.getMinExpectedNumber() > 0:
        # 1. RL Agent Logic
        current_phase = traci.trafficlight.getPhase(tls_id)
        
        # Track Phase Change for Logging
        if current_phase != last_phase:
            if last_phase != -1:
                duration_sec = (step - phase_start_step) * 0.45
                print(f"Phase {last_phase} -> {current_phase} | Duration: {duration_sec:.2f}s")
            last_phase = current_phase
            phase_start_step = step
            
        # RL Prediction
        lanes = [
            ("N_to_J0_0", "N_to_J0_1"),
            ("E_to_J0_0", "E_to_J0_1"),
            ("S_to_J0_0", "S_to_J0_1"),
            ("W_to_J0_0", "W_to_J0_1")
        ]
        queues = []
        counts = []
        for l0, l1 in lanes:
            queues.append(traci.lane.getLastStepHaltingNumber(l0) + traci.lane.getLastStepHaltingNumber(l1))
            counts.append(traci.lane.getLastStepVehicleNumber(l0) + traci.lane.getLastStepVehicleNumber(l1))
            
        obs = np.array([current_phase] + queues + counts, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        
        # NOTE: WE MUST REPLICATE THE ENV CONSTRAINT HERE IF WE WANT THE AGENT TO BEHAVE IDENTICALLY?
        # OR DOES THE AGENT LEARN IT? 
        # The agent learns it, BUT in run scripts we are the "Env".
        # We must enforce the constraint too, otherwise the visual demo will differ from training!
        # Wait, usually we just run the Env wrapper. But here we are manually stepping Traci.
        # So we MUST Implement the logic here for visual fidelity.
        
        time_since_change = (step - phase_start_step) * 0.45
        min_green = 10.0
        
        if action == 1:
            if time_since_change < min_green:
                # CONSTRAINT: IGNORE SWITCH
                pass 
            else:
                 # EXECUTE SWITCH
                 # Cycle logic: +1 (Env handles yellow, here we emulate simple switch for now or careful transition)
                 # Let's match Env step logic: Switch to Yellow then Green.
                 # For manual run script, simpler to just setPhase(next).
                 # Wait, Env does yellow transition.
                 # If we just setPhase(current+1), we rely on SUMO or logic.
                 # Let's simple toggle to next phase index in logic.
                 traci.trafficlight.setPhase(tls_id, (current_phase + 1) % 8)
                 
        traci.simulationStep()
        step += 1
            
    traci.close()

if __name__ == "__main__":
    run_stable_demo()
