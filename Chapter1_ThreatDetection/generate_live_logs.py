"""
This script collects live system metrics and writes them to a CSV file
for testing a threat detection system.
"""

from __future__ import annotations

import csv
import os
import socket
import time
from datetime import UTC, datetime
import signal
import sys
from pathlib import Path

import psutil

BASE_DIR = Path(__file__).resolve().parent


def signal_handler(sig, frame):
    """Handle graceful shutdown on Ctrl+C."""
    print("\n[!] Stopping logger safely...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_top_process() -> tuple[int, str]:
    """
    Return the PID and name of the top CPU-consuming process.

    Returns:
        tuple: (pid, process_name)
    """
    top_pid = 0
    top_name = "unknown"
    top_cpu = -1.0

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            cpu = proc.cpu_percent(interval=None)
            if cpu > top_cpu:
                top_cpu = cpu
                top_pid = proc.info["pid"] or 0
                top_name = proc.info["name"] or "unknown"
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return top_pid, top_name


def warm_up_process_cpu() -> None:
    """
    Warm up per-process CPU counters so cpu_percent(interval=None)
    returns useful values on the next read.
    """
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def get_system_metrics(previous_net: dict[str, int] | None = None) -> tuple[dict, dict]:
    """
    Collect live system metrics.

    Returns:
        metrics: Dictionary of current system metrics
        current_net: Latest network counters for the next calculation
    """
    timestamp = datetime.now(UTC).isoformat()

    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()

    process_id, process_name = get_top_process()

    current_net = {
        "bytes_sent": net.bytes_sent,
        "bytes_recv": net.bytes_recv,
    }

    if previous_net is None:
        network_in = 0
        network_out = 0
    else:
        network_in = current_net["bytes_recv"] - previous_net["bytes_recv"]
        network_out = current_net["bytes_sent"] - previous_net["bytes_sent"]

    metrics = {
        "timestamp": timestamp,
        "hostname": socket.gethostname(),
        "process_id": process_id,
        "process_name": process_name,
        "cpu_usage": round(cpu_usage, 2),
        "memory_usage": round(memory.percent, 2),
        "disk_usage": round(disk.percent, 2),
        "network_in_bytes": int(network_in),
        "network_out_bytes": int(network_out),
        "process_count": len(psutil.pids()),
        "boot_time": datetime.fromtimestamp(psutil.boot_time(), UTC).isoformat(),
    }

    return metrics, current_net


def write_header_if_needed(filename: str, fieldnames: list[str]) -> None:
    """Create CSV header if the file does not exist yet."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


def generate_live_logs(filename: str = BASE_DIR / "live_system_logs.csv", interval: int = 5) -> None:
    """
    Continuously collect live system metrics and append them to a CSV file.

    Args:
        filename: Output CSV filename
        interval: Seconds between samples
    """
    previous_net = None

    # Warm up process CPU counters before first real sample
    warm_up_process_cpu()

    sample, previous_net = get_system_metrics(previous_net)
    fieldnames = list(sample.keys())
    write_header_if_needed(filename, fieldnames)

    print(f"[+] Writing live system logs to {filename}")
    print(f"[+] Sampling every {interval} seconds. Press Ctrl+C to stop.")

    try:
        while True:
            log, previous_net = get_system_metrics(previous_net)

            with open(filename, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(log)

            print(
                f"[{log['timestamp']}] "
                f"PID={log['process_id']} | "
                f"PROC={log['process_name']} | "
                f"CPU={log['cpu_usage']}% | "
                f"MEM={log['memory_usage']}% | "
                f"DISK={log['disk_usage']}% | "
                f"NET_IN={log['network_in_bytes']}B | "
                f"NET_OUT={log['network_out_bytes']}B | "
                f"PROCS={log['process_count']}"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[!] Logging stopped by user.")


if __name__ == "__main__":
    generate_live_logs()
