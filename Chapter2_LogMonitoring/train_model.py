"""
Train an Isolation Forest model for anomaly detection on log data.
The script loads CSV log data, preprocesses it into numeric features,
trains the model, and saves the model artifact.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Create a base directory path for loading the log data
BASE_DIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "log_stream.csv"
MODEL_PATH = BASE_DIR / "isolation_forest_model.pkl"

# Load the log data
log_data = pd.read_csv(CSV_PATH)

if log_data.empty:
    raise ValueError(f"Input CSV is empty: {CSV_PATH}")

# Convert all columns to numeric model features
# (handles categorical/text fields common in logs)
X = pd.get_dummies(log_data, dummy_na=True)

# Clean problematic values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

if X.shape[1] == 0:
    raise ValueError("No usable features found after preprocessing.")

# Train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
model.fit(X)

# Save both model and feature schema for consistent inference later
artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
}
joblib.dump(artifact, MODEL_PATH)

print(f"[>] Model trained and saved successfully: {MODEL_PATH}")
