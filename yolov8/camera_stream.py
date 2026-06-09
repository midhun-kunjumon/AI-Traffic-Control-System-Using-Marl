import threading
import time
import cv2
import urllib.request
import numpy as np

# CAMERA_URLS maps each direction's camera to its ESP32 camera streaming endpoint
CAMERA_URLS = {
    "North": "http://192.168.1.3",
    "East": "http://192.168.1.6",
    "South": "http://192.168.1.7",
    "West": "http://192.168.1.5",
}

class CameraManager:
    """Manages multithreaded frame capture streams from multiple ESP32 cameras."""
    def __init__(self, camera_urls=None, mode="snapshot"):
        self.urls = camera_urls or CAMERA_URLS
        self.mode = mode # "snapshot" or "mjpeg"
        self.frames = {d: None for d in self.urls.keys()}
        self.status = {d: {"online": False, "last_frame_age": 999.0, "url": url} for d, url in self.urls.items()}
        self.running = False
        self.threads = []
        
    def start_all(self):
        self.running = True
        for direction, url in self.urls.items():
            t = threading.Thread(target=self._capture_loop, args=(direction, url), daemon=True)
            self.threads.append(t)
            t.start()
            
    def _capture_loop(self, direction, url):
        while self.running:
            frame = self._grab_frame(url, direction)
            if frame is not None:
                self.frames[direction] = frame
                self.status[direction]["online"] = True
                self.status[direction]["last_frame_age"] = 0.0
            else:
                self.status[direction]["online"] = False
                self.status[direction]["last_frame_age"] += 1.0
            time.sleep(1.0) # Capture every second
            
    def _grab_frame(self, url, direction, timeout=5):
        # Try capture endpoints
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
        return None
        
    def get_all_frames(self):
        return dict(self.frames)
        
    def get_status(self):
        return dict(self.status)
        
    def stop_all(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
        self.threads.clear()
