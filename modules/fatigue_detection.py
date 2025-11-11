import cv2
import numpy as np
import threading
import time

class EyeShiftMonitor:
    def __init__(self, camera=None):
        self.camera = camera
        self.running = False
        self.movement_freq = 0
        self.last_position = None
        self.shifts = 0

    def start(self):
        if self.camera is None:
            raise ValueError("CameraManager not provided to EyeShiftMonitor.")
        self.running = True
        self.thread = threading.Thread(target=self._track_shifts, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join()

    def _track_shifts(self):
        print("[INFO] EyeShiftMonitor started using shared camera...")
        start_time = time.time()
        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml").detectMultiScale(gray, 1.1, 4)
            if len(eyes) > 0:
                x, y, ew, eh = eyes[0]
                cx, cy = x + ew // 2, y + eh // 2
                if self.last_position is not None:
                    dx, dy = abs(cx - self.last_position[0]), abs(cy - self.last_position[1])
                    if dx > 10 or dy > 10:
                        self.shifts += 1
                self.last_position = (cx, cy)

            if time.time() - start_time >= 10:
                self.movement_freq = self.shifts / 10.0
                self.shifts = 0
                start_time = time.time()

    def get_movement_frequency(self):
        return round(self.movement_freq, 2)
