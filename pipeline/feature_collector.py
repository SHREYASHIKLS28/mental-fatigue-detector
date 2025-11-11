# pipeline/feature_collector.py
import threading
import time
import pandas as pd
import os
from datetime import datetime

from modules.basic import EyeBlinkMonitor
from modules.fatigue_detection import EyeShiftMonitor
from modules.keystroke_monitor import KeystrokeMonitor
from modules.tab_switch_monitor import TabSwitchMonitor
from modules.camera import CameraManager


class FeatureCollector:
    def __init__(self, model_path=None):
        print("🎥 Initializing shared camera...")
        self.camera = CameraManager()

        # Initialize all monitoring modules
        self.eye_monitor = EyeBlinkMonitor(camera=self.camera)
        self.eyeball_tracker = EyeShiftMonitor(camera=self.camera)
        self.keystroke_monitor = KeystrokeMonitor()
        self.tab_monitor = TabSwitchMonitor()

        # Start continuous threads
        self.eye_monitor.start()
        self.eyeball_tracker.start()
        self.tab_monitor.start()
        self.keystroke_monitor.start()

        self.model_path = model_path
        self.data = []
        self.lock = threading.Lock()
        self.is_running = False
        self.autosave_interval = 60  # auto-save every 60 seconds

        print("[DEBUG] FeatureCollector initialized.")

    # ---------------------------------------------------------------------- #
    def collect_features(self):
        """Collect current metrics from all modules."""
        features = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            # Eye-based fatigue metrics
            **self.eye_monitor.metrics,

            # Eyeball movement
            "eyeball_movement": self.eyeball_tracker.get_movement_frequency()
            if hasattr(self.eyeball_tracker, "get_movement_frequency") else 0,

            # Keystroke metrics
            "keystroke_speed": self.keystroke_monitor.get_typing_speed()
            if hasattr(self.keystroke_monitor, "get_typing_speed") else 0,

            "error_rate": self.keystroke_monitor.get_error_rate()
            if hasattr(self.keystroke_monitor, "get_error_rate") else 0,

            # Tab switching metrics
            "tab_switch_count": self.tab_monitor.metrics.get("total_switches", 0),
            "focus_ratio": self.tab_monitor.metrics.get("focus_ratio", 0),
            "avg_focus_duration": self.tab_monitor.metrics.get("avg_focus_duration", 0),
            "avg_distracted_duration": self.tab_monitor.metrics.get("avg_distracted_duration", 0),
            "total_distracted_time": self.tab_monitor.metrics.get("total_distracted_time", 0),
        }

        print("[LIVE DATA]", features)
        return features

    # ---------------------------------------------------------------------- #
    def start_collection(self, interval=10):
        """Continuously collect data and make fatigue predictions."""
        from fatigue_inference import predict_fatigue  # import here to avoid circular imports

        if self.is_running:
            return
        self.is_running = True

        def run():
            print(f"[INFO] Collecting features every {interval}s...\n")
            while self.is_running:
                # 1️⃣ Collect real-time feature snapshot
                features = self.collect_features()

                # 2️⃣ Predict fatigue level using trained LightGBM model
                try:
                    fatigue_level = predict_fatigue(features)
                    features["predicted_fatigue_level"] = fatigue_level
                    print(f"[PREDICTION] Fatigue Level → {fatigue_level}")
                except Exception as e:
                    print(f"[WARN] Prediction failed: {e}")
                    features["predicted_fatigue_level"] = "UNKNOWN"

                # 3️⃣ Store features (with prediction) in dataset
                with self.lock:
                    self.data.append(features)

                # 4️⃣ Wait for next cycle
                time.sleep(interval)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------------- #
    def stop_collection(self):
        """Gracefully stop all modules and save data."""
        print("[INFO] Stopping all monitors and saving data...")
        self.is_running = False

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join()

        # Stop all running modules safely
        try:
            self.eye_monitor.stop()
            self.eyeball_tracker.stop()
            self.tab_monitor.stop()
        except Exception as e:
            print(f"[WARN] Error stopping modules: {e}")

        # Release camera
        try:
            self.camera.stop()
        except Exception:
            pass

        # Final save
        self.save_to_csv("data/output/realtime_fatigue_dataset.csv")
        print("[INFO] All systems stopped successfully.")

    # ---------------------------------------------------------------------- #
    def save_to_csv(self, filename="data/output/realtime_fatigue_dataset.csv"):
        """Appends collected data to the existing CSV file instead of overwriting."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with self.lock:
            if not self.data:
                print("[WARN] No data collected yet — nothing to save.")
                return
            df = pd.DataFrame(self.data)

        try:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                df.to_csv(filename, mode='a', header=False, index=False)
                print(f"[INFO] Data appended to existing file: {filename}")
            else:
                df.to_csv(filename, mode='w', header=True, index=False)
                print(f"[INFO] New dataset created at: {filename}")
        except PermissionError:
            print(f"[ERROR] Permission denied for '{filename}'. "
                  f"Close the file if it's open in Excel or VS Code and try again.")
        except Exception as e:
            print(f"[ERROR] Failed to save data: {e}")
