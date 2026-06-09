import os
from ultralytics import YOLO

class VehicleDetector:
    """YOLOv8 vehicle detection wrapper with tracking for queue and speed estimation."""
    def __init__(self, model_path=None, conf=0.3):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "yolov8", "best1.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, "yolov8", "best.pt")
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.track_history = {} # For tracking and speed estimation
        
    def detect(self, frame, camera_id=None):
        if frame is None:
            return self._empty_result()
            
        # Run YOLOv8 tracking (persist=True for tracking across frames)
        results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)
        result = results[0]
        
        # Count vehicles and estimate queue/movement
        vehicle_count = 0
        moving_count = 0
        queue_length = 0
        total_speed = 0.0
        
        boxes = result.boxes if result.boxes is not None else result.obb
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id].lower()
                
                # Check confidence
                conf_val = float(box.conf[0])
                if conf_val < self.conf:
                    continue
                
                vehicle_count += 1
                
                # Check speed or movement if tracking id is available
                if box.id is not None:
                    track_id = int(box.id[0])
                    xyxy = box.xyxy[0].tolist()
                    center = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                    
                    if camera_id not in self.track_history:
                        self.track_history[camera_id] = {}
                        
                    if track_id in self.track_history[camera_id]:
                        prev_center = self.track_history[camera_id][track_id]
                        # Distance in pixels
                        dist = ((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)**0.5
                        # Assume 5fps or standard interval to estimate pixel speed
                        speed = dist
                        total_speed += speed
                        if speed > 2.0: # Moving if threshold exceeded
                            moving_count += 1
                        else:
                            queue_length += 1
                    else:
                        # First time seeing this vehicle
                        queue_length += 1
                    
                    self.track_history[camera_id][track_id] = center
                else:
                    # Fallback if no tracking id (non-moving/queue by default)
                    queue_length += 1
                    
        avg_speed = total_speed / max(1, vehicle_count)
        
        return {
            "vehicle_count": vehicle_count,
            "queue_length": queue_length,
            "moving_count": moving_count,
            "avg_speed_estimate": avg_speed,
        }
        
    def _empty_result(self):
        return {
            "vehicle_count": 0,
            "queue_length": 0,
            "moving_count": 0,
            "avg_speed_estimate": 0.0,
        }
        
    def reset_tracking(self):
        self.track_history.clear()
