"""This script detects anomalies in the synthetic logs generated for testing the threat detection system."""
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the synthetic logs


def load_logs(filename='synthetic_logs.csv'):
    df = pd.read_csv(filename)
    print(f"[^] Loaded {len(df)} logs from {filename}")
    return df

# Create and fit the Isolation Forest model to detect anomalies


def detect_anomalies(df):
    features = df[['cpu_usage', 'memory_usage', 'network_in', 'network_out']]
    model = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = model.fit_predict(features)
    anomalies = df[df['anomaly'] == -1]
    print(f"[^] Detected {len(anomalies)} anomalies:")
    print(anomalies)


# Create the Isolation Forest model to detect anomalies
model = IsolationForest(
    n_estimators=100,   # Number of trees in the forest
    contamination=0.02,  # Set contamination to 2% to account for the injected anomalies
    random_state=42     # Set random state for reproducibility
)

# Load the synthetic logs
logs_df = load_logs()

# # Sanity Check - Print the columns of the loaded DataFrame to verify structure
# print(logs_df.columns.tolist())

# Fit the model to the features
features = logs_df[['cpu_usage', 'memory_usage', 'network_in', 'network_out']]
model.fit(features)

# Predict anomalies
logs_df['anomaly'] = model.predict(features)

# Filter and display the detected anomalies
anomalies = logs_df[logs_df['anomaly'] == -1]

# Print the detected anomalies
print(f"[!] Detected {len(anomalies)} anomalies:\n")

# Display the anomalies with relevant details
print(anomalies[['cpu_usage',
      'memory_usage', 'network_in', 'network_out']])

# Save the detected anomalies to a CSV file for further analysis
anomalies.to_csv('detected_anomalies.csv', index=False)
print("\n[>] Detected anomalies saved to detected_anomalies.csv")
