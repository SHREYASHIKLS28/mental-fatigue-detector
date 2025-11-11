# modules/basic.py
import cv2
import dlib
import time
import os
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
import threading


class EyeBlinkMonitor:
    def __init__(self, camera=None, duration=60, calibration_time=3, output_csv=None):
        self.camera = camera
        self.duration = duration
        self.calibration_time = calibration_time
        self.output_csv = output_csv or "data/raw/eye_blink_microexp.csv"

        # Load dlib's face landmark predictor
        shape_predictor_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "models",
            "shape_predictor_68_face_landmarks.dat"
        )
        shape_predictor_path = os.path.abspath(shape_predictor_path)
        if not os.path.exists(shape_predictor_path):
            raise FileNotFoundError(f"Model not found: {shape_predictor_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.lm_model = dlib.shape_predictor(shape_predictor_path)
        (self.L_start, self.L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.R_start, self.R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # State tracking
        self.blink_count = 0
        self.blink_durations = []
        self.eye_closed_frames = deque(maxlen=1500)
        self.calibration_ears = []
        self.running = False

        # Initialize metrics
        self.metrics = {
            "blinks_per_min": 0,
            "avg_blink_duration": 0,
            "perclos": 0,
            "brow_raise": 0,
            "lip_distance": 0,
            "eye_openness": 0,
            "fatigue_level": "UNKNOWN"
        }

    # ------------------------------------------------------------------ #
    def EAR_cal(self, eye):
        """Compute Eye Aspect Ratio."""
        v1 = dist.euclidean(eye[1], eye[5])
        v2 = dist.euclidean(eye[2], eye[4])
        h1 = dist.euclidean(eye[0], eye[3])
        return (v1 + v2) / (2.0 * h1)

    # ------------------------------------------------------------------ #
    def start(self):
        if self.camera is None:
            raise ValueError("CameraManager instance not provided to EyeBlinkMonitor.")
        self.running = True
        self.thread = threading.Thread(target=self._process_frames, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the monitor gracefully."""
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=2)
        print("[INFO] EyeBlinkMonitor stopped.")

    # ------------------------------------------------------------------ #
    def _process_frames(self):
        """Continuously process frames from shared camera."""
        print("[INFO] EyeBlinkMonitor started using shared camera...")

        start_time = time.time()
        eyes_closed = False
        blink_start_time = None
        blink_thresh_low, blink_thresh_high = None, None
        self.start_time = time.time()

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) == 0:
                # Record “no face detected” as open eyes (neutral)
                self.eye_closed_frames.append(0)
                continue

            for face in faces:
                shape = self.lm_model(gray, face)
                shape = face_utils.shape_to_np(shape)
                lefteye = shape[self.L_start:self.L_end]
                righteye = shape[self.R_start:self.R_end]
                left_EAR = self.EAR_cal(lefteye)
                right_EAR = self.EAR_cal(righteye)
                avg_EAR = (left_EAR + right_EAR) / 2

                # Calibration phase
                if time.time() - start_time <= self.calibration_time:
                    self.calibration_ears.append(avg_EAR)
                    continue

                if not blink_thresh_low:
                    mean_EAR = sum(self.calibration_ears) / len(self.calibration_ears)
                    blink_thresh_low = mean_EAR * 0.85
                    blink_thresh_high = mean_EAR * 1.05
                    print(f"[INFO] Calibrated EAR range: {blink_thresh_low:.3f}-{blink_thresh_high:.3f}")

                # Blink detection logic
                if avg_EAR < blink_thresh_low and not eyes_closed:
                    eyes_closed = True
                    blink_start_time = time.time()
                elif avg_EAR > blink_thresh_high and eyes_closed:
                    eyes_closed = False
                    self.blink_count += 1
                    if blink_start_time:
                        blink_duration = time.time() - blink_start_time
                        self.blink_durations.append(blink_duration)

                # Microexpressions
                brow_left, brow_right, nose_top = shape[21], shape[22], shape[27]
                lips_top, lips_bottom = shape[62], shape[66]
                brow_raise = dist.euclidean(brow_left, nose_top) + dist.euclidean(brow_right, nose_top)
                lip_distance = dist.euclidean(lips_top, lips_bottom)

                # Append closed/open status
                self.eye_closed_frames.append(1 if eyes_closed else 0)
                eye_openness = avg_EAR

                # Update fatigue metrics
                self._update_metrics(brow_raise, lip_distance, eye_openness)

            # Display overlay
            cv2.putText(frame, f"Fatigue: {self.metrics['fatigue_level']}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks/min: {self.metrics['blinks_per_min']}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Fatigue Monitor (Shared Camera)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    def _update_metrics(self, brow_raise, lip_distance, eye_openness):
        """Compute fatigue-related metrics."""
        duration = max(time.time() - self.start_time, 1)
        avg_blink_duration = (
            sum(self.blink_durations) / len(self.blink_durations)
            if self.blink_durations else 0
        )

        # Compute PERCLOS (Percentage of time eyes closed)
        perclos = (
            (sum(self.eye_closed_frames) / len(self.eye_closed_frames)) * 100
            if self.eye_closed_frames else 0
        )
        blinks_per_min = self.blink_count * (60 / duration)

        # Fatigue classification
        if blinks_per_min > 12 and perclos < 10 and avg_blink_duration < 0.25:
            fatigue = "ACTIVE"
        elif (8 <= blinks_per_min <= 12) or (10 <= perclos < 20):
            fatigue = "MILD"
        elif (5 <= blinks_per_min < 8) or (20 <= perclos < 30) or (avg_blink_duration >= 0.25):
            fatigue = "FATIGUE"
        else:
            fatigue = "SEVERE"

        self.metrics.update({
            "blinks_per_min": round(blinks_per_min, 2),
            "avg_blink_duration": round(avg_blink_duration, 3),
            "perclos": round(perclos, 2),
            "brow_raise": round(brow_raise, 3),
            "lip_distance": round(lip_distance, 3),
            "eye_openness": round(eye_openness, 3),
            "fatigue_level": fatigue
        })
