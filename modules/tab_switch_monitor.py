# modules/tab_switch_monitor.py
import time
import threading
import pygetwindow as gw
import csv
import os
from datetime import datetime

class TabSwitchMonitor:
    def __init__(self,
                 check_interval=1.0,
                 output_csv="data/raw/tab_switch_log.csv",
                 allowed_keywords=None):
        self.check_interval = check_interval
        self.output_csv = output_csv
        self.allowed_keywords = [k.lower() for k in (allowed_keywords or [
            "chrome", "firefox", "edge", "vscode", "pycharm", "jupyter", "notepad", "word", "excel"
        ])]

        self.current_window = None
        self.tab_switch_count = 0
        self.focus_time = 0
        self.distracted_time = 0
        self.running = False
        self.thread = None
        self.start_time = None
        self.last_switch_time = None
        self.last_title = None

        self.metrics = {
            "focus_ratio": 0,
            "total_switches": 0,
            "avg_focus_duration": 0,
            "avg_distracted_duration": 0,
            "total_distracted_time": 0,
            "longest_distraction": 0
        }

    def start(self):
        """Start monitoring in a background thread."""
        if self.running:
            print("[WARN] TabSwitchMonitor already running.")
            return
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        self.running = True
        self.thread = threading.Thread(target=self._monitor_tabs, daemon=True)
        self.thread.start()
        print("[INFO] TabSwitchMonitor started...")

    def stop(self):
        """Stop the monitor and save data."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self._save_to_csv()
        print("[INFO] TabSwitchMonitor stopped and data saved.")

    def _monitor_tabs(self):
        """Track active window title and detect tab switches."""
        self.start_time = datetime.now()
        self.last_switch_time = time.time()
        self.last_title = self._get_active_window()

        while self.running:
            active_title = self._get_active_window()
            now = time.time()
            if active_title != self.last_title:
                duration = now - self.last_switch_time
                if self.last_title and self._is_allowed(self.last_title):
                    self.focus_time += duration
                else:
                    self.distracted_time += duration

                self.tab_switch_count += 1
                self.last_title = active_title
                self.last_switch_time = now

                print(f"[TAB SWITCH] Active window: {active_title}")
                self._update_metrics()

            time.sleep(self.check_interval)

    def _get_active_window(self):
        try:
            w = gw.getActiveWindow()
            if w:
                return w.title
        except Exception:
            return None
        return None

    def _is_allowed(self, title):
        if not title:
            return False
        t = title.lower()
        return any(kw in t for kw in self.allowed_keywords)

    def _update_metrics(self):
        total_time = (time.time() - self.start_time.timestamp()) if self.start_time else 1
        total_focus_time = self.focus_time
        total_distracted_time = self.distracted_time
        total_switches = self.tab_switch_count

        focus_ratio = total_focus_time / total_time if total_time else 0
        avg_focus_duration = total_focus_time / total_switches if total_switches else 0
        avg_distracted_duration = total_distracted_time / total_switches if total_switches else 0
        longest_distraction = max(avg_distracted_duration, self.metrics["longest_distraction"])

        self.metrics.update({
            "focus_ratio": round(focus_ratio, 3),
            "total_switches": total_switches,
            "avg_focus_duration": round(avg_focus_duration, 2),
            "avg_distracted_duration": round(avg_distracted_duration, 2),
            "total_distracted_time": round(total_distracted_time, 2),
            "longest_distraction": round(longest_distraction, 2)
        })

    def _save_to_csv(self):
        """Save summary metrics to CSV."""
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics.keys())
            writer.writerow(self.metrics.values())

    def get_features(self):
        """Return live tab monitoring metrics."""
        return self.metrics
