import traci

class EmergencyManager:
    def __init__(self, tls_id="J0", detection_range=300):
        self.tls_id = tls_id
        self.detection_range = detection_range
        
        # Lanes entering the intersection (2 lanes per edge now)
        # N_to_J0_0 (Left), N_to_J0_1 (Straight/Right)
        # We need to scan ALL of them.
        self.incoming_lanes = [
            "N_to_J0_0", "N_to_J0_1",
            "E_to_J0_0", "E_to_J0_1",
            "S_to_J0_0", "S_to_J0_1",
            "W_to_J0_0", "W_to_J0_1"
        ]
        
        # Map lanes to Green Phase indices (Split Phase Logic)
        # Phase 0: North Green
        # Phase 2: East Green
        # Phase 4: South Green
        # Phase 6: West Green
        self.lane_to_phase = {
            "N_to_J0_0": 0, "N_to_J0_1": 0,
            "E_to_J0_0": 2, "E_to_J0_1": 2,
            "S_to_J0_0": 4, "S_to_J0_1": 4,
            "W_to_J0_0": 6, "W_to_J0_1": 6
        }

    def check_emergency(self):
        """
        Scans all incoming lanes for emergency vehicles.
        Returns:
            active_ev_lane (str or None): The lane ID where EV is approaching.
        """
        for lane in self.incoming_lanes:
            # Get list of vehicle IDs on the lane
            vehs = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehs:
                v_class = traci.vehicle.getVehicleClass(veh_id)
                # Also check typeID just in case vClass isn't set perfectly in xml
                v_type = traci.vehicle.getTypeID(veh_id)
                
                if v_class == "emergency" or "emergency" in v_type:
                    # Check distance to junction
                    dist = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(veh_id)
                    
                    if dist <= self.detection_range:
                        return lane
        return None

    def get_override_phase(self, ev_lane):
        """
        Returns the target Green phase for the given lane.
        """
        return self.lane_to_phase.get(ev_lane, 0)

