
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore", category=UserWarning)


# Load dataset
DATA_PATH = "data/output/realtime_fatigue_dataset.csv"

print("🔍 [INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
print(f"[INFO] Dataset loaded with shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}")

# Drop unwanted columns if they exist
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"], errors="ignore")

# Remove duplicates and NaNs
df = df.drop_duplicates().dropna()

# Keep only numeric columns
# Keep all numeric columns + target column (if present)
target_col = "fatigue_level"
if target_col in df.columns:
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    df = df[numeric_cols + [target_col]]
else:
    raise ValueError("'fatigue_level' column not found in dataset!")

print(f"[INFO] Cleaned dataset shape: {df.shape}")


# Encode target label
if "fatigue_level" not in df.columns:
    raise ValueError("'fatigue_level' column not found in dataset!")

encoder = LabelEncoder()
df["fatigue_level"] = encoder.fit_transform(df["fatigue_level"])

print(f"[INFO] Encoded target classes: {list(encoder.classes_)}")


# Train/test split (handle rare classes safely)
df = df.sample(frac=0.8, random_state=42)  # use 80% of data
X = df.drop(columns=["fatigue_level"])
y = df["fatigue_level"]

# Class count
label_counts = y.value_counts()
print("\n[INFO] Class distribution before split:")
print(label_counts)

# Drop rare classes
rare_classes = label_counts[label_counts < 2].index.tolist()
if rare_classes:
    print(f"\n[WARN] Dropping rare classes (too few samples): {rare_classes}")
    df = df[~df["fatigue_level"].isin(rare_classes)]
    X = df.drop(columns=["fatigue_level"])
    y = df["fatigue_level"]

# Split safely
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add slight noise to simulate real-world variation
noise_factor = 0.02  # try 0.01–0.05 range
X_train_scaled = X_train_scaled + np.random.normal(0, noise_factor, X_train_scaled.shape)

noise_factor = 0.08
X_train_noisy = X_train_scaled + np.random.normal(0, noise_factor, X_train_scaled.shape)

# Light dropout effect (5% feature masking)
mask = np.random.binomial(1, 0.95, X_train_noisy.shape)
X_train_noisy = X_train_noisy * mask

# Use 75% of data for training to preserve diversity
X_train_noisy, _, y_train_small, _ = train_test_split(
    X_train_noisy, y_train, test_size=0.25, random_state=42
)

# LightGBM tuned for moderate bias-variance balance
model = LGBMClassifier(
    n_estimators=150,           # More trees for stability
    learning_rate=0.07,         # Balanced learning rate
    num_leaves=24,              # Moderate complexity
    max_depth=6,
    min_data_in_leaf=40,
    feature_fraction=0.65,      # Randomly drop 35% of features
    bagging_fraction=0.7,       # Randomly drop 30% of data
    bagging_freq=6,
    reg_alpha=0.3,              # Moderate L1 regularization
    reg_lambda=0.4,             # Moderate L2 regularization
    subsample_for_bin=100000,
    random_state=42,
    verbose=-1
)

model.fit(X_train_noisy, y_train_small)

# Evaluate model
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"\nAccuracy: {acc * 100:.2f}%")
print(f"Weighted F1-score: {f1:.2f}")

#Show only labels in test set
labels_in_test = sorted(y_test.unique())
target_names = [encoder.classes_[i] for i in labels_in_test]
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))


# Confusion Matrix Visualization
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Convert to DataFrame for readability
cm_df = pd.DataFrame(cm, 
                     index=target_names, 
                     columns=target_names)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix — Fatigue Level Classification")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1_weighted")
print(f"\nMean Accuracy (CV): {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}")
print(f"Mean F1-Score (CV): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}\n")


# Feature importance visualization
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10], palette="viridis")
plt.title("Top 10 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()

# Save model, scaler, encoder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fatigue_model.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

