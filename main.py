# main.py
import time
import os
from pipeline.feature_collector import FeatureCollector
from fatigue_inference import predict_fatigue


def main():
    print("=== REAL-TIME MENTAL FATIGUE MONITORING PIPELINE ===")
    print("[INFO] Initializing all modules...")

    # Initialize pipeline collector (this starts the shared camera + monitors)
    try:
        collector = FeatureCollector(model_path=None)
    except Exception as e:
        print(f"[ERROR] Failed to initialize modules: {e}")
        return

    print("[INFO] Starting feature collection...")
    collector.start_collection(interval=10)

    print("\n[INFO] Press 'Q' in the OpenCV window or 'Ctrl + C' in the terminal to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected — shutting down...")

    finally:
        # Ensure proper shutdown and dataset saving
        try:
            collector.stop_collection()
            save_path = os.path.join("data", "output", "realtime_fatigue_dataset.csv")
            collector.save_to_csv(save_path)
            print(f"[INFO] Data successfully saved to {save_path}")
        except Exception as e:
            print(f"[WARN] Error during shutdown: {e}")

        print("[INFO] Pipeline stopped. Exiting now.")


if __name__ == "__main__":
    main()
