"""
OPTAD – Mock Data Generator for Threat Detection Model
Generates realistic training (benign baseline) and testing (mixed benign, suspicious, malicious) datasets.
"""

import os
import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Benign Process Templates for Training ──────────────────────────────────────
BENIGN_PROCESS_PROFILES = [
    # (name, cpu_mean, cpu_std, mem_mean, mem_std, threads_mean, conn_mean, child_mean)
    ("systemd",        0.1,  0.05,  15,   3,    1,   0, 0),
    ("launchd",        0.2,  0.1,   20,   5,    2,   0, 0),
    ("WindowServer",   4.5,  2.0,   180,  40,   12,  0, 0),
    ("explorer.exe",   1.5,  0.8,   95,   20,   8,   1, 1),
    ("chrome",         3.8,  2.5,   220,  60,   18,  4, 2),
    ("code",           2.5,  1.5,   280,  50,   14,  2, 1),
    ("python3",        1.2,  1.0,   65,   20,   4,   0, 0),
    ("bash",           0.0,  0.05,  8,    2,    1,   0, 0),
    ("zsh",            0.0,  0.05,  12,   3,    1,   0, 0),
    ("sshd",           0.1,  0.05,  18,   4,    2,   1, 0),
    ("dock",           0.5,  0.2,   55,   10,   5,   0, 0),
    ("slack",          2.0,  1.2,   310,  70,   16,  5, 0),
    ("spotify",        1.8,  0.9,   140,  30,   10,  3, 0),
    ("node",           1.5,  1.0,   110,  35,   8,   2, 0),
    ("git",            0.4,  0.3,   25,   8,    2,   0, 0),
    ("finder",         0.6,  0.3,   85,   15,   6,   0, 0),
    ("loginwindow",    0.2,  0.1,   45,   10,   4,   0, 0),
    ("dbus-daemon",    0.1,  0.05,  10,   2,    1,   0, 0),
    ("mysqld",         3.0,  1.5,   250,  50,   22,  6, 0),
    ("nginx",          0.8,  0.4,   40,   10,   4,   8, 2),
]

def generate_train_data(n_samples=600):
    rows = []
    pid_counter = 1000

    for _ in range(n_samples):
        prof = random.choice(BENIGN_PROCESS_PROFILES)
        name, cpu_m, cpu_s, mem_m, mem_s, thr_m, conn_m, child_m = prof
        
        # Sample realistic values with non-negative bounds
        cpu = max(0.0, round(float(np.random.normal(cpu_m, cpu_s)), 2))
        mem = max(5.0, round(float(np.random.normal(mem_m, mem_s)), 2))
        thr = max(1, int(round(np.random.normal(thr_m, max(1, thr_m * 0.25)))))
        conn = max(0, int(round(np.random.poisson(conn_m))))
        child = max(0, int(round(np.random.poisson(child_m))))
        normal_folder = 1.0  # Training baseline is strictly legitimate / normal paths

        rows.append({
            "process_name": name,
            "pid": pid_counter,
            "cpu_usage": cpu,
            "memory_mb": mem,
            "thread_count": thr,
            "connections": conn,
            "children": child,
            "normal_folder": normal_folder
        })
        pid_counter += random.randint(1, 5)

    df = pd.DataFrame(rows)
    return df


def generate_test_data():
    rows = []
    pid_counter = 5000

    # 1. Benign Samples (60 samples)
    for _ in range(60):
        prof = random.choice(BENIGN_PROCESS_PROFILES)
        name, cpu_m, cpu_s, mem_m, mem_s, thr_m, conn_m, child_m = prof
        cpu = max(0.0, round(float(np.random.normal(cpu_m, cpu_s)), 2))
        mem = max(5.0, round(float(np.random.normal(mem_m, mem_s)), 2))
        thr = max(1, int(round(np.random.normal(thr_m, max(1, thr_m * 0.25)))))
        conn = max(0, int(round(np.random.poisson(conn_m))))
        child = max(0, int(round(np.random.poisson(child_m))))

        rows.append({
            "process_name": name,
            "pid": pid_counter,
            "cpu_usage": cpu,
            "memory_mb": mem,
            "thread_count": thr,
            "connections": conn,
            "children": child,
            "normal_folder": 1.0,
            "expected_label": "Benign",
            "threat_type": "Normal System/User Activity"
        })
        pid_counter += random.randint(2, 6)

    # 2. Suspicious Samples (20 samples: moderate anomalies, weird folders, or localized spikes)
    suspicious_scenarios = [
        ("dev_server",       14.5, 420.0, 16, 12, 2, 1.0, "High local connections and memory usage"),
        ("data_backup.sh",   28.0, 180.0, 5,  1,  3, 1.0, "Unusual CPU spike on shell script"),
        ("python_scraper",   18.2, 350.0, 12, 18, 1, 1.0, "High external HTTP connection count"),
        ("electron_app",     12.0, 580.0, 24, 8,  2, 1.0, "Heavier than normal memory footprint"),
        ("updater_tool",     5.0,  95.0,  6,  4,  1, 0.0, "Running from user Downloads folder (normal_folder=0)"),
        ("test_runner",      32.0, 290.0, 10, 0,  4, 1.0, "Bursty compilation CPU activity"),
        ("media_converter",  42.0, 310.0, 18, 0,  1, 1.0, "FFmpeg intensive encoding spike"),
        ("temp_cli",         2.5,  45.0,  3,  2,  0, 0.0, "Binary executing from temporary workspace"),
        ("sync_agent",       8.5,  210.0, 14, 15, 0, 1.0, "Cloud sync burst traffic"),
        ("p2p_client",       15.0, 260.0, 20, 22, 1, 1.0, "Elevated peer network connections")
    ]
    for _ in range(2):
        for name, cpu, mem, thr, conn, child, norm_f, desc in suspicious_scenarios:
            rows.append({
                "process_name": name,
                "pid": pid_counter,
                "cpu_usage": cpu,
                "memory_mb": mem,
                "thread_count": thr,
                "connections": conn,
                "children": child,
                "normal_folder": norm_f,
                "expected_label": "Suspicious",
                "threat_type": desc
            })
            pid_counter += random.randint(2, 7)

    # 3. Anomaly / Malicious Samples (20 samples: critical threats, severe deviations)
    malicious_scenarios = [
        ("xmrig_miner",       98.5, 850.0,  48, 4,   0,  0.0, "Cryptominer: maxed CPU, disguised binary in /tmp"),
        ("revshell_sh",       0.5,  12.0,   1,  1,   4,  0.0, "Reverse shell: hidden spawn in /dev/shm with child shells"),
        ("ransomware_enc",    89.0, 620.0,  32, 0,   0,  0.0, "Ransomware: bulk file encryption from /tmp with high IO/CPU"),
        ("c2_beacon",         1.2,  45.0,   4,  45,  0,  0.0, "C2 Beacon: abnormal connection burst from non-standard path"),
        ("syn_flooder",       78.0, 95.0,   12, 160, 0,  0.0, "DDoS flooder: anomalous socket connections"),
        ("fork_bomb",         94.0, 310.0,  64, 0,   55, 0.0, "Denial-of-Service: rapid process fork bombing"),
        ("mem_dumper",        24.0, 1850.0, 8,  0,   0,  0.0, "Credential dumping: extreme memory scan anomaly"),
        ("stealth_keylogger", 0.1,  18.0,   2,  8,   0,  0.0, "Spyware: stealth keystroke logging hooked in hidden dir"),
        ("dropper_payload",   45.0, 210.0,  15, 25,  8,  0.0, "Malware dropper: downloading stage2 payloads from /tmp"),
        ("data_exfiltrator",  35.0, 480.0,  16, 75,  1,  0.0, "Data exfiltration: high-volume outbound socket transfers")
    ]
    for _ in range(2):
        for name, cpu, mem, thr, conn, child, norm_f, desc in malicious_scenarios:
            rows.append({
                "process_name": name,
                "pid": pid_counter,
                "cpu_usage": cpu,
                "memory_mb": mem,
                "thread_count": thr,
                "connections": conn,
                "children": child,
                "normal_folder": norm_f,
                "expected_label": "Anomaly",
                "threat_type": desc
            })
            pid_counter += random.randint(2, 7)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    train_df = generate_train_data(600)
    train_path = os.path.join(DATA_DIR, "train_data.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Generated {len(train_df)} training samples -> {train_path}")

    test_df = generate_test_data()
    test_path = os.path.join(DATA_DIR, "test_data.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Generated {len(test_df)} testing samples -> {test_path}")
