"""This script generates synthetic logs for testing the threat detection system."""
import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)

# Generate synthetic system log data


def generate_logs(filename='synthetic_logs.csv', num_logs=500):
    logs = []
    for _ in range(num_logs):
        log = {
            'cpu_usage': round(random.uniform(0, 100), 2),
            'memory_usage': round(random.uniform(0, 100), 2),
            'network_in': round(random.uniform(0, 1000), 2),
            'network_out': round(random.uniform(0, 1000), 2),
        }
        logs.append(log)

    # Inject some anomalies
    for _ in range(10):
        log = {
            'cpu_usage': random.uniform(90, 100),
            'memory_usage': random.uniform(90, 100),
            'network_in': random.uniform(2000, 3000),
            'network_out': random.uniform(2000, 3000),
        }
        logs.append(log)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(logs)
    df.to_csv(filename, index=False)
    print(f"[_] Saved {len(df)} synthetic logs to {filename}")


if __name__ == "__main__":
    generate_logs()
