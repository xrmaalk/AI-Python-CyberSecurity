"""
Detect anomalies from live system log data using Isolation Forest.

This version ignores non-training fields such as:
- timestamp
- hostname
- process_id
- process_name
- boot_time

But includes process fields in anomaly output for investigation.

Expected input file:
- live_system_logs.csv

Expected useful columns:
- cpu_usage
- memory_usage
- network_in
- network_out

Ignored columns:
- timestamp
- hostname
- any other non-numeric columns unless explicitly selected
"""

from __future__ import annotations
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_FILE = BASE_DIR / "live_system_logs.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "detected_anomalies.csv"


def load_logs(filename: str = DEFAULT_INPUT_FILE) -> pd.DataFrame:
    """Load logs from CSV."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Log file not found: {filename}")

    df = pd.read_csv(filename)
    print(f"[+] Loaded {len(df)} logs from {filename}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare numeric features for anomaly detection.

    Excludes timestamp and process identity fields from training.
    """
    working_df = df.copy()

    ignored_columns = [
        "timestamp",
        "hostname",
        "process_id",
        "process_name",
        "boot_time",
    ]

    working_df = working_df.drop(
        columns=[col for col in ignored_columns if col in working_df.columns],
        errors="ignore",
    )

    selected_cols = [
        "cpu_usage",
        "memory_usage",
        "disk_usage",
        "network_in_bytes",
        "network_out_bytes",
        "process_count",
    ]

    missing = [col for col in selected_cols if col not in working_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    features = working_df[selected_cols].copy()

    for col in selected_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    valid_mask = features.notna().all(axis=1)
    cleaned_features = features.loc[valid_mask].copy()
    cleaned_df = df.loc[valid_mask].copy()

    if cleaned_features.empty:
        raise ValueError("No valid rows remain after cleaning the input data.")

    print(f"[+] Using features: {selected_cols}")
    print(
        f"[+] Retained {len(cleaned_features)} valid rows for training/detection")

    return cleaned_df, cleaned_features, selected_cols


def detect_anomalies(
    df: pd.DataFrame,
    features: pd.DataFrame,
    contamination: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train Isolation Forest and return scored results and anomalies only.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(features)

    predictions = pipeline.predict(features)
    scores = pipeline.decision_function(features)

    results_df = df.copy()
    results_df["anomaly"] = predictions
    results_df["anomaly_score"] = scores

    anomalies = results_df[results_df["anomaly"] == -1].copy()

    print(f"[!] Detected {len(anomalies)} anomalies")
    return results_df, anomalies


def save_anomalies(anomalies: pd.DataFrame, filename: str = DEFAULT_OUTPUT_FILE) -> None:
    """Save anomalies to CSV."""
    anomalies.to_csv(filename, index=False)
    print(f"[>] Detected anomalies saved to {filename}")


def main() -> None:
    """Main function to load logs, prepare features, detect anomalies, and save results."""
    df = load_logs()
    cleaned_df, features, selected_cols = prepare_features(df)
    _, anomalies = detect_anomalies(cleaned_df, features, contamination=0.02)

    if anomalies.empty:
        print("\n[+] No anomalies detected.")
    else:
        preferred_cols = [
            "timestamp",
            "hostname",
            "process_id",
            "process_name",
            "cpu_usage",
            "memory_usage",
            "disk_usage",
            "network_in_bytes",
            "network_out_bytes",
            "process_count",
            "anomaly_score",
        ]
        display_cols = [
            col for col in preferred_cols if col in anomalies.columns]

        print("\n[!] Anomalies found:\n")
        print(anomalies[display_cols].to_string(index=False))

    save_anomalies(anomalies)


if __name__ == "__main__":
    main()
