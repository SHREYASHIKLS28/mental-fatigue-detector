# modules/keystroke_monitor.py
import time
import csv
from pynput import keyboard
import threading
import os
from collections import deque


class KeystrokeMonitor:
    def __init__(self, output_csv="data/raw/keystroke_log.csv", window_size=60):
        
        
        self.output_csv = output_csv
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        self.window_size = window_size
        self.keystroke_log = deque(maxlen=2000)
        self.hold_times = {}
        self.backspace_count = 0
        self.listener = None
        self.running = False
        self.lock = threading.Lock()

        self.metrics = {
            "keystroke_speed": 0,  # keys/min
            "avg_hold_time": 0,
            "avg_pause": 0,
            "error_rate": 0        # %
        }

        self.total_keys_pressed = 0
        self.start_time = time.time()

    # ------------------------------------------------ #
    def _on_press(self, key):
        now = time.time()
        with self.lock:
            if key == keyboard.Key.backspace:
                self.backspace_count += 1

            self.hold_times[key] = now
            self.keystroke_log.append({"time": now, "key": str(key), "event": "press"})
            self.total_keys_pressed += 1

        # Update metrics instantly
        self._update_metrics()

    def _on_release(self, key):
        now = time.time()
        with self.lock:
            if key in self.hold_times:
                duration = now - self.hold_times[key]
                self.keystroke_log.append({
                    "time": now,
                    "key": str(key),
                    "event": "release",
                    "hold_time": round(duration, 3)
                })
                del self.hold_times[key]

    # ------------------------------------------------ #
    def _update_metrics(self):
        now = time.time()
        with self.lock:
            recent = [k for k in self.keystroke_log if now - k["time"] <= self.window_size]
            presses = [k for k in recent if k["event"] == "press"]

            keys_per_min = (len(presses) / self.window_size) * 60 if self.window_size > 0 else 0

            hold_times = [k["hold_time"] for k in recent if "hold_time" in k]
            avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

            press_times = [k["time"] for k in presses]
            pauses = [t2 - t1 for t1, t2 in zip(press_times[:-1], press_times[1:])]
            avg_pause = sum(pauses) / len(pauses) if pauses else 0

            error_rate = (self.backspace_count / self.total_keys_pressed * 100
                          if self.total_keys_pressed > 0 else 0)

            self.metrics.update({
                "keystroke_speed": round(keys_per_min, 2),
                "avg_hold_time": round(avg_hold_time, 3),
                "avg_pause": round(avg_pause, 3),
                "error_rate": round(error_rate, 2)
            })

    # ------------------------------------------------ #
    def start(self):
        """Start global keyboard listener."""
        if self.running:
            print("[WARN] KeystrokeMonitor already running.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_listener, daemon=True)
        self.thread.start()
        print("[INFO] KeystrokeMonitor started (global mode).")

    def _run_listener(self):
        # Run as a global listener (non-blocking)
        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False  # must be False, otherwise system input gets blocked
        ) as listener:
            self.listener = listener
            listener.join()

    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()
        self._save_to_csv()
        print("[INFO] KeystrokeMonitor stopped and data saved.")

    # ------------------------------------------------ #
    def get_typing_speed(self):
        self._update_metrics()
        return self.metrics["keystroke_speed"]

    def get_error_rate(self):
        self._update_metrics()
        return self.metrics["error_rate"]

    def get_features(self):
        self._update_metrics()
        return self.metrics

    # ------------------------------------------------ #
    def _save_to_csv(self):
        with self.lock:
            if not self.keystroke_log:
                return
            with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "key", "event", "hold_time"])
                writer.writeheader()
                writer.writerows(self.keystroke_log)
