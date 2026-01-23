import os
import sys
import traci
import sumolib
import time

import os
import sys
import traci
import sumolib
import time

# Since emergency_manager is now in the same directory (traci/), we can import directly
# provided we run as a module or from the correct CWD.
# But for script execution:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from emergency_manager import EmergencyManager

def run_stage6():
    # Paths
    base_path = os.getcwd() # Assumes run from root
    # If run from traci folder, we need adjustment. Let's assume run from root c:\ai-traffic-control
    net_file = os.path.join(base_path, "sumo", "stage6_emergency", "intersection.net.xml")
    route_file = os.path.join(base_path, "sumo", "stage6_emergency", "routes.rou.xml")
    
    sumo_binary = sumolib.checkBinary('sumo-gui') # Use GUI to verify
    cmd = [
        sumo_binary,
        "-n", net_file,
        "-r", route_file,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "--step-length", "0.5",
        "--start",
        "--quit-on-end"
    ]
    
    print("Starting Stage 6 Validation...")
    traci.start(cmd)
    
    manager = EmergencyManager()
    tls_id = "J0"
    
    step = 0
    # Standard Fixed Time cycle for background: 
    # 0(G, 30s) -> 1(Y, 4s) -> 2(G, 30s) -> 3(Y, 4s)
    # We implement a basic fixed cycle manually to have a 'baseline' to override
    cycle_timer = 0
    current_fixed_phase = 0
    # Durations for 0, 1, 2, 3
    durations = [30, 4, 30, 4] 
    
    # Track when we entered the current phase (for yellow safety)
    phase_start_step = 0
    last_phase = -1
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        
        # 1. Check Emergency
        ev_lane = manager.check_emergency()
        
        current_phase = traci.trafficlight.getPhase(tls_id)
        if current_phase != last_phase:
            phase_start_step = step
            last_phase = current_phase
            
        time_in_phase = (step - phase_start_step) * 0.5 # seconds
        
        if ev_lane:
            # --- EMERGENCY OVERRIDE LOGIC ---
            target_phase = manager.get_override_phase(ev_lane)
            
            print(f"Step {step}: EMERGENCY DETECTED on {ev_lane}. Target: {target_phase}. Current: {current_phase}")
            
            if current_phase == target_phase:
                # We are in correct Green. HOLD IT.
                traci.trafficlight.setPhase(tls_id, target_phase)
                # Reset cycle timer so fixed usage doesn't snap back immediately after
                cycle_timer = 0 
                
            else:
                # We need to switch
                # Logic:
                # G -> Y -> G_target
                
                # If conflicting Green (0 or 2)
                if current_phase % 2 == 0:
                     # Check Min Green (e.g. 5s) safety?
                     if time_in_phase >= 5.0:
                         # Switch to Yellow
                         print("   -> Switching to Yellow")
                         traci.trafficlight.setPhase(tls_id, current_phase + 1)
                     else:
                         print("   -> Waiting for Min Green")
                         
                # If Yellow (1 or 3)
                elif current_phase % 2 == 1:
                     # Check Min Yellow (e.g. 3s)
                     if time_in_phase >= 3.0:
                         # Switch to Target Green
                         # Helper: (1->2, 3->0)
                         next_theoretical = (current_phase + 1) % 4
                         if next_theoretical == target_phase:
                             print("   -> Switching to Target Green")
                             traci.trafficlight.setPhase(tls_id, target_phase)
                         else:
                             # This shouldn't happen in 2-phase system, but just in case
                             traci.trafficlight.setPhase(tls_id, next_theoretical)
                     else:
                         print("   -> Waiting for Min Yellow")

        else:
            # --- NORMAL CONTROL (Fixed Time) ---
            # Update cycle
            cycle_timer += 0.5
            
            required_duration = durations[current_fixed_phase]
            
            # If we were overridden, we need to resync our internal state?
            # Or just blindly apply the phase?
            # Issue: If EV held Green for 20s, and we were at t=10 of Green, should we resume at t=10 or t=30?
            # User said: "Resume from its previous internal state".
            # So if we were interrupted at t=10, we resume trying to finish that phase?
            # BUT, the physical light might be in a different phase now.
            
            # Use 'Standard' recovery:
            # If physical != planned:
            #   We must transition physical -> planned.
            
            # Let's simplify:
            # Just increment timer. If timer > duration, move next.
            # AND force the light to match 'current_fixed_phase'.
            
            if cycle_timer >= required_duration:
                cycle_timer = 0
                current_fixed_phase = (current_fixed_phase + 1) % 4
                
            # Now apply 'current_fixed_phase' to Traffic Light
            # BUT we must respect safety transitions if we are far apart.
            # (e.g. EV left us at Phase 0, but Fixed wants Phase 2).
            # We can't jump 0->2.
            
            # Simple Controller Wrapper:
            # If actual != desired:
            #   If actual is Green and desired is Diff Green: Go Yellow.
            #   If actual is Yellow: Wait.
            
            desired = current_fixed_phase
            
            if current_phase == desired:
                traci.trafficlight.setPhase(tls_id, desired)
            else:
                # Transition Logic
                if current_phase % 2 == 0: # Green
                    if time_in_phase > 5: # Min Green
                        traci.trafficlight.setPhase(tls_id, current_phase + 1)
                elif current_phase % 2 == 1: # Yellow
                    if time_in_phase > 3:
                         traci.trafficlight.setPhase(tls_id, (current_phase + 1) % 4)

    traci.close()

if __name__ == "__main__":
    run_stage6()
