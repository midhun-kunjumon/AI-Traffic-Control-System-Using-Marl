import os
import sys
import optparse
import traci
from sumolib import checkBinary

# --- Configuration ---
# Path to the SUMO configuration from this script's location
# Script is in root/traci/
# Config is in root/sumo/stage4_traci_control/
SUMO_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sumo", "stage4_traci_control")
SUMO_CFG_FILE = os.path.join(SUMO_CONFIG_DIR, "intersection.sumocfg")
GUI_MODE = True  # Set to False for command-line only (faster training)

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def get_lane_stats(lane_id):
    """Retrieves real-time statistics for a specific lane."""
    # Number of vehicles on the lane in the last step
    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
    
    # Waiting time: sum of waiting time of all vehicles on the lane (seconds)
    # A vehicle is waiting if its speed is below 0.1m/s
    waiting_time = traci.lane.getWaitingTime(lane_id)
    
    # Queue length: Number of vehicles with speed < 0.1 m/s
    # getLastStepHaltingNumber returns number of vehicles with speed < 0.1 m/s
    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
    
    return {
        "vehicle_count": vehicle_count,
        "waiting_time": waiting_time,
        "queue_length": queue_length
    }

def print_stats(step, lane_ids):
    print(f"\n--- Step {step} ---")
    print(f"Traffic Light Phase: {traci.trafficlight.getPhase('J0')} (Duration: {traci.trafficlight.getPhaseDuration('J0')})")
    for lane in lane_ids:
        stats = get_lane_stats(lane)
        if stats["vehicle_count"] > 0: # Only print active lanes to reduce clutter
            print(f"Lane {lane}: {stats}")

def run():
    step = 0
    # Get all lane IDs controlled by traffic light 'J0'
    # This is a bit advanced, for now we manually list the incoming lanes we care about based on the net.xml
    # Incoming lanes: N_to_J0_0, E_to_J0_0, S_to_J0_0, W_to_J0_0
    monitored_lanes = ["N_to_J0_0", "E_to_J0_0", "S_to_J0_0", "W_to_J0_0"]
    
    # We can also get them programmatically:
    # tls_id = "J0"
    # lanes = traci.trafficlight.getControlledLanes(tls_id)
    # monitored_lanes = list(set(lanes)) # Remove duplicates

    print("Starting simulation...")
    
    # Logic to switch phases programmatically
    # Standard Cycle: 
    # Phase 0: NS Green (42s)
    # Phase 1: NS Yellow (3s)
    # Phase 2: EW Green (42s)
    # Phase 3: EW Yellow (3s)
    
    # We will override the default program logic
    # Let's say we want to switch every 30 seconds for demonstration
    cycle_time = 30 
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Real-time data retrieval
        if step % 10 == 0: # Print stats every 10 steps to not spam console
            print_stats(step, monitored_lanes)
        
        # --- Programmatic Control Example ---
        # NOTE: When using TraCI to set phase, the duration counter in SUMO might behave differently 
        # than fixed-time. It's often better to check the time since last change.
        
        # Simple Logic: Alternate Green every 300 steps (Wait, simulation step is usually 1s? Let's check)
        # Assuming step length is 1s (default).
        
        current_phase = traci.trafficlight.getPhase("J0")
        
        # This is a basic demonstration of forcing a phase change.
        # In a real Cycle based system, you would track time_since_last_change.
        
        # Let's say we want a faster cycle than the config file (which was 42s)
        # We will force a switch every 10 seconds just to prove we have control.
        # The logic below overrides the automatic timer.
        
        # Important: traci.trafficlight.setPhaseDuration(tlsID, duration) extends the current phase.
        # traci.trafficlight.setPhase(tlsID, index) jumps to a new phase immediately.
        
        # Let's just monitor for now. The user said "Programmatically control... instead of fixed-time".
        # So we will forcefully switch.
        
        if step % 200 == 100: # At step 100, 300, 500...
             print("Forcefully switching to EW Green (Phase 2)")
             traci.trafficlight.setPhase("J0", 2) # Switch to EW Green
        elif step % 200 == 0 and step > 0: # At step 200, 400...
             print("Forcefully switching to NS Green (Phase 0)")
             traci.trafficlight.setPhase("J0", 0) # Switch to NS Green

        step += 1

    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    options = get_options()

    # Determine which binary to use
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # Start TraCI
    # The config path is already absolute based on our configuration above
    traci.start([sumoBinary, "-c", SUMO_CFG_FILE])
    run()
