# fatigue_inference.py
import joblib
import numpy as np
import pandas as pd

# Load trained components
model = joblib.load("models/fatigue_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")


def predict_fatigue(features_dict):
    """
    Takes the latest features collected from your monitors
    and predicts the fatigue level using LightGBM.
    """
    df = pd.DataFrame([features_dict])

    # Drop extra columns that were not used during training
    drop_cols = ["timestamp", "fatigue_level", "predicted_fatigue_level"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Scale features using the trained scaler
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)


    # Predict fatigue label
    pred = model.predict(df_scaled)
    fatigue_label = encoder.inverse_transform(pred)[0]

    # (Optional) confidence score
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_scaled).max() * 100
        print(f"[PREDICTION] Fatigue Level → {fatigue_label} ({prob:.2f}% confidence)")
    else:
        print(f"[PREDICTION] Fatigue Level → {fatigue_label}")

    return fatigue_label


# Example test
if __name__ == "__main__":
    sample = {
        "blinks_per_min": 10.2,
        "avg_blink_duration": 0.22,
        "perclos": 15.4,
        "brow_raise": 1.12,
        "lip_distance": 0.43,
        "eye_openness": 0.23,
        "eyeball_movement": 0.7,
        "keystroke_speed": 3.4,
        "error_rate": 0.05,
        "tab_switch_count": 4,
        "focus_ratio": 0.8,
        "avg_focus_duration": 12.5,
        "avg_distracted_duration": 3.4,
        "total_distracted_time": 30.0
    }
    print("Predicted Fatigue Level:", predict_fatigue(sample))
