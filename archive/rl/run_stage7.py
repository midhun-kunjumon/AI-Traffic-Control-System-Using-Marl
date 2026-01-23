import os
import sys
import traci
import sumolib
import time

# Since emergency_manager is now in the same directory (rl/), we can import directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from emergency_manager import EmergencyManager

def run_stage8():
    # Paths for Stage 8 (Asymmetric)
    base_path = os.getcwd()
    cfg_file = os.path.join(base_path, "sumo", "stage8_asymmetric", "intersection.sumocfg")
    
    # Run with GUI to visually verify
    sumo_binary = sumolib.checkBinary('sumo-gui') 
    cmd = [
        sumo_binary,
        "-c", cfg_file,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "--start",
        "--quit-on-end"
    ]
    
    print("Starting Stage 8 (Asymmetric Traffic) Validation...")
    traci.start(cmd)
    
    manager = EmergencyManager(tls_id="J0")
    
    step = 0
    # Split Phase Cycle: N(30)->Y(3)->E(30)->Y(3)->S(30)->Y(3)->W(30)->Y(3)
    # Total 8 phases in program: 0, 1, 2, 3, 4, 5, 6, 7
    durations = [30, 3, 30, 3, 30, 3, 30, 3] 
    current_fixed_phase = 0
    cycle_timer = 0
    
    last_phase = -1
    phase_start_step = 0
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        
        current_phase = traci.trafficlight.getPhase("J0")
        if current_phase != last_phase:
            phase_start_step = step
            last_phase = current_phase
        time_in_phase = (step - phase_start_step) * 0.5
        
        # 1. EMERGENCY CHECK
        ev_lane = manager.check_emergency()
        
        if ev_lane:
            target_phase = manager.get_override_phase(ev_lane)
            if step % 10 == 0:
                print(f"Step {step}: EMERGENCY on {ev_lane}. Target={target_phase}, Current={current_phase}")
            
            if current_phase == target_phase:
                traci.trafficlight.setPhase("J0", target_phase)
                cycle_timer = 0
            else:
                # Transition Logic for 8-phase system
                # If even (Green): Check Min Green -> Switch to Yellow (next phase)
                if current_phase % 2 == 0: 
                     if time_in_phase >= 5.0:
                         traci.trafficlight.setPhase("J0", current_phase + 1)
                # If odd (Yellow): Check Min Yellow -> Switch to Next
                elif current_phase % 2 == 1: 
                     if time_in_phase >= 3.0:
                         next_ph = (current_phase + 1) % 8
                         # Optimize: if next_ph leads to target, let it go.
                         # If target is far away, we just cycle through.
                         # Advanced: Skip phases? 
                         # For now, just cycle fast.
                         traci.trafficlight.setPhase("J0", next_ph)
        else:
            # 2. NORMAL FIXED TIME
            cycle_timer += 0.5
            if cycle_timer >= durations[current_fixed_phase]:
                cycle_timer = 0
                current_fixed_phase = (current_fixed_phase + 1) % 8
                
            desired = current_fixed_phase
            if current_phase != desired:
                # Simple transition helper
                if current_phase % 2 == 0 and time_in_phase > 5:
                    traci.trafficlight.setPhase("J0", current_phase + 1)
                elif current_phase % 2 == 1 and time_in_phase > 3:
                     traci.trafficlight.setPhase("J0", (current_phase + 1) % 8)
            else:
                traci.trafficlight.setPhase("J0", desired)

    traci.close()

if __name__ == "__main__":
    run_stage7()
