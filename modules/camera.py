import cv2
import threading
import time

class CameraManager:
    def __init__(self, cam_index=0):
        self.cam_index = cam_index
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self._start_camera()

    def _start_camera(self):
        """Start the webcam and background thread safely."""
        print(f"Initializing camera at index {self.cam_index}...")
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Please check your camera or restart it.")

        # Warm up camera
        time.sleep(0.3)
        self.running = True

        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print("CameraManager started successfully.")

    def _update(self):
        """Continuously grab frames until stopped."""
        while self.running:
            if not self.cap.isOpened():
                print("Camera disconnected — attempting reconnect...")
                time.sleep(1)
                self._restart_camera()
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            with self.lock:
                self.frame = frame

        # Cleanup after stop
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("Camera thread stopped cleanly.")

    def _restart_camera(self):
        """Restart camera safely after disconnection."""
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        time.sleep(0.5)
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)

    def get_frame(self):
        """Return the most recent frame (safe copy)."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop the background thread and release camera."""
        if not self.running:
            return

        print("Stopping CameraManager...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        if self.cap and self.cap.isOpened():
            self.cap.release()

        try:
            cv2.destroyAllWindows()
        except:
            pass

        print("Camera released and stopped successfully.")