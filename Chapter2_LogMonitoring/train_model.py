"""
This script trains an Isolation Forest model for anomaly detection on log data. It reads the log data from a CSV file, preprocesses it, and then fits the model. Finally, it saves the trained model to a file for later use in anomaly detection.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import pathlib

# Create a base directory path for loading the log data
BASE_DIR = pathlib.Path(__file__).resolve().parent

# Load the log data
log_data = pd.read_csv(BASE_DIR / 'log_stream.csv')

# Train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
model.fit(log_data)

# Save the trained model to a file
joblib.dump(model, BASE_DIR / 'isolation_forest_model.pkl')
print("[>]Model trained and saved successfully.")
