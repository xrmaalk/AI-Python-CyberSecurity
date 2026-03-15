"""Append new logs to the CSV file."""
from datetime import datetime
from time import sleep
import pathlib
import pandas as pd

# Define the path to the log file
BASE_DIR = pathlib.Path(__file__).parent.resolve()
LOG_FILE = BASE_DIR / 'log_stream.csv'

# Function to append a new log entry to the CSV file
def append_log(log_message):
    """Append a new log entry to the CSV file."""
    timestamp = datetime.now().isoformat()
    new_entry = pd.DataFrame({"timestamp": [timestamp], "log_message": [log_message]})

    # Append to the CSV file, creating it if it doesn't exist    
    if LOG_FILE.exists():
        new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(LOG_FILE, index=False)

if __name__ == "__main__":
    # Example log messages to append
    example_logs = [
        "User login successful for user 'admin'",
        "Failed password attempt for user 'guest'",
        "File 'confidential.txt' accessed by user 'john_doe'",
        "Unexpected error occurred in module 'payment_processor'",
        "User 'alice' logged out",
    ]

    for log in example_logs:
        append_log(log)
        print(f"Appended log: {log}")

        sleep(1)  # Simulate delay between log entries

