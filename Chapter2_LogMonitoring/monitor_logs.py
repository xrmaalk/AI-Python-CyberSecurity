"""Monitor logs for anomalies using a pre-trained Isolation Forest model."""

import pathlib
import os
import time
import joblib
import pandas as pd
from colorama import init, Fore, Style

BASE_DIR = pathlib.Path(__file__).parent.resolve()
LOG_FILE = BASE_DIR / 'log_stream.csv'
MODEL_FILE = BASE_DIR / 'isolation_forest_model.pkl'
POLL_INTERVAL = 2  # seconds

# Initialize colorama
init(autoreset=True)

def load_model():
    """Load the pre-trained Isolation Forest model and feature schema."""
    artifact = joblib.load(MODEL_FILE)
    # Support both old (bare model) and new (artifact dict) formats
    if isinstance(artifact, dict):
        return artifact["model"], artifact["feature_columns"]
    return artifact, None

def load_logs():
    """Load logs from the CSV file."""
    return pd.read_csv(LOG_FILE)



def check_new_logs(model, feature_columns, last_seen):
    """Check for new logs and predict anomalies."""
    df = load_logs()
    if len(df) <= last_seen:
        return last_seen  # No new logs

    new_logs = df.iloc[last_seen:].reset_index(drop=True)  # reset index to 0-based

    # Preprocess to match training features
    X = pd.get_dummies(
        new_logs.drop(columns=["timestamp", "log_message"], errors="ignore"),
        dummy_na=True,
    )

    # Align columns to training schema
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)

    predictions = model.predict(X)

    for idx, pred in enumerate(predictions):
        # Safely get log_message or fall back to raw row
        row = new_logs.iloc[idx]
        log_message = row.get("log_message", str(row.to_dict()))

        if pred == -1:  # Anomaly detected
            print(f"{Fore.RED}[ANOMALY] {log_message}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}[NORMAL]  {log_message}{Style.RESET_ALL}")

    return len(df)

def monitor_logs():
    """Continuously monitor logs for anomalies."""
    model, feature_columns = load_model()
    last_seen = 0
    print("Monitoring logs for anomalies...")
    while True:
        try:
            last_seen = check_new_logs(model, feature_columns, last_seen)
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n[<|'_'|>] Monitor stopped by user.")
            break
        except Exception as e:
            print(f"{Fore.YELLOW}Error: {e}{Style.RESET_ALL}")
            time.sleep(POLL_INTERVAL)



if __name__ == "__main__":
    """Start monitoring logs for anomalies."""
    monitor_logs()

