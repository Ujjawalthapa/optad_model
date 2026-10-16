"""
OPTAD / OS Sentinel – Machine Learning Process Threat Detection
===============================================================
A complete, streamlined ML model pipeline for process-level threat detection.

Pipeline Steps:
  1. Load Data   : Loads train (normal baseline) and test (mixed threats) datasets
  2. Clean Data  : Validates numeric types, handles NaNs/infinities, removes duplicates
  3. EDA         : Summary statistics, correlations, and distribution plots
  4. Train Model : Fits IsolationForest on normal process behavior
  5. Evaluate    : Tests model against Benign, Suspicious, and Anomaly processes
  6. Save Model  : Saves trained model bundle for deployment

Run:
  python ai/train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ── Visualization Setup ───────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "eda")

TRAIN_FILE  = os.path.join(DATA_DIR, "train_data.csv")
TEST_FILE   = os.path.join(DATA_DIR, "test_data.csv")
MODEL_FILE  = os.path.join(OUTPUT_DIR, "model.pkl")
PRED_FILE   = os.path.join(OUTPUT_DIR, "test_predictions.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Feature Definitions ───────────────────────────────────────────────────────
FEATURES = [
    "cpu_usage",      # CPU percentage [0.0 - 100.0]
    "memory_mb",      # Resident memory in megabytes
    "thread_count",   # Total active threads
    "connections",    # Open network sockets
    "children",       # Number of spawned child processes
    "normal_folder",  # 1.0 = standard OS folder, 0.0 = suspicious/temp folder
]

# ── Model Configuration ───────────────────────────────────────────────────────
CONTAMINATION    = 0.01  # Expected fraction of anomalies in clean baseline
N_ESTIMATORS     = 100   # Number of isolation trees
RANDOM_STATE     = 42    # Reproducibility seed
SUSPICIOUS_SCORE = 40    # Threshold for Suspicious classification (40-60)
ANOMALY_SCORE    = 60    # Threshold for Anomaly classification (>= 60)


# =============================================================================
# STEP 1 – Load Data
# =============================================================================
def load_datasets():
    print(f"\n{'='*70}")
    print("STEP 1: Loading Datasets")
    print(f"{'='*70}")

    if not os.path.isfile(TRAIN_FILE) or not os.path.isfile(TEST_FILE):
        print("Dataset not found. Generating fresh mock data...")
        from ai.generate_mock_data import generate_train_data, generate_test_data
        train_df = generate_train_data(600)
        train_df.to_csv(TRAIN_FILE, index=False)
        test_df = generate_test_data()
        test_df.to_csv(TEST_FILE, index=False)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    print(f"Training dataset : {len(train_df)} rows  ({TRAIN_FILE})")
    print(f"Testing dataset  : {len(test_df)} rows  ({TEST_FILE})")
    print(f"Features used    : {FEATURES}")

    return train_df, test_df


# =============================================================================
# STEP 2 – Clean Data
# =============================================================================
def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    label_str = "Training" if is_training else "Testing"
    print(f"\n{'='*70}")
    print(f"STEP 2: Cleaning Data ({label_str})")
    print(f"{'='*70}")

    initial_count = len(df)
    
    # 1. Ensure numeric types for ML features
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. Check and report missing values
    missing = df[FEATURES].isna().sum()
    if missing.sum() > 0:
        print("Found missing values:\n", missing[missing > 0].to_string())
        # Impute with column medians
        for col in FEATURES:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val if not pd.isna(median_val) else 0.0, inplace=True)
    else:
        print(f"Missing values check: 0 missing entries in feature columns.")

    # 4. Remove exact duplicates in training data
    if is_training:
        dup_count = df.duplicated(subset=FEATURES).sum()
        if dup_count > 0:
            df.drop_duplicates(subset=FEATURES, inplace=True)
            print(f"Removed {dup_count} duplicate rows.")
        else:
            print("Duplicate rows check: 0 duplicates.")

    print(f"Cleaned {label_str} data: {len(df)} rows ready.")
    return df


# =============================================================================
# STEP 3 – Exploratory Data Analysis (EDA)
# =============================================================================
def run_eda(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("STEP 3: Exploratory Data Analysis (EDA)")
    print(f"{'='*70}")

    # Summary statistics
    stats = df[FEATURES].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T
    stats["median"] = df[FEATURES].median()
    print("\n--- Summary Statistics (Normal Baseline) ---")
    print(stats[["min", "median", "mean", "max", "std", "99%"]].round(2).to_string())

    # Correlation matrix
    corr = df[FEATURES].corr(numeric_only=True)
    print("\n--- Feature Correlation Matrix ---")
    print(corr.round(3).to_string())

    # Plots
    if not PLOTS_AVAILABLE:
        print("\nNote: matplotlib/seaborn not found. Skipping plot generation.")
        return

    # 1. Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), FEATURES):
        ax.hist(df[col], bins=25, color="#2563EB", edgecolor="white", alpha=0.85)
        ax.set_title(f"Distribution: {col}", fontweight="bold", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("OPTAD Process Baseline Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    hist_path = os.path.join(PLOT_DIR, "1_histograms.png")
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)

    # 2. Box Plots (Outlier analysis)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), FEATURES):
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor="#93C5FD", color="#1E40AF"),
                   medianprops=dict(color="#DC2626", linewidth=2))
        ax.set_title(f"Spread & Outliers: {col}", fontweight="bold", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("Process Features Spread & Range Bounds", fontsize=14, fontweight="bold")
    plt.tight_layout()
    box_path = os.path.join(PLOT_DIR, "2_boxplots.png")
    fig.savefig(box_path, dpi=120)
    plt.close(fig)

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Process Feature Correlations", fontsize=12, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(PLOT_DIR, "3_correlation.png")
    fig.savefig(corr_path, dpi=120)
    plt.close(fig)

    print(f"\nSaved EDA plots to '{PLOT_DIR}':")
    print(f"  - 1_histograms.png")
    print(f"  - 2_boxplots.png")
    print(f"  - 3_correlation.png")


# =============================================================================
# STEP 4 – Train Isolation Forest
# =============================================================================
def train_model(train_df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("STEP 4: Model Training")
    print(f"{'='*70}")

    X_train = train_df[FEATURES].values.astype(float)

    # Record 99th-percentile limits & feature medians for reference
    limits = {feat: float(np.percentile(X_train[:, i], 99)) for i, feat in enumerate(FEATURES)}
    medians = {feat: float(np.median(X_train[:, i])) for i, feat in enumerate(FEATURES)}

    print(f"Training IsolationForest on {len(X_train)} baseline samples:")
    print(f"  - n_estimators  : {N_ESTIMATORS}")
    print(f"  - contamination : {CONTAMINATION}")
    print(f"  - random_state  : {RANDOM_STATE}")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )
    model.fit(X_train)
    print("Training complete!")

    return model, limits, medians


# =============================================================================
# STEP 5 – Scoring & Evaluation
# =============================================================================
def compute_risk_score(model, X: np.ndarray, limits: dict) -> np.ndarray:
    """
    Computes calibrated [0 - 100] risk score combining:
      1. Machine Learning Anomaly Score (Isolation Forest decision function)
      2. Feature Deviations relative to 99th percentile baseline limits
      3. Execution Environment / Folder Legitimacy (normal_folder)
    """
    decision = model.decision_function(X)
    forest_risk = 100.0 / (1.0 + np.exp(8.0 * decision))

    scores = []
    normal_idx = FEATURES.index("normal_folder")

    for i, row in enumerate(X):
        r_ml = forest_risk[i]
        is_normal_folder = (row[normal_idx] >= 0.5)

        # Check deviations beyond baseline 99th percentile limits
        ratio_sum = 0.0
        for feat_idx, feat in enumerate(FEATURES):
            if feat == "normal_folder":
                continue
            val = row[feat_idx]
            lim = limits.get(feat, 1.0)
            if val > lim:
                ratio_sum += (val - lim) / max(lim, 1.0)

        # Base score from ML IsolationForest + deviation penalty
        base_score = r_ml * 0.70 + min(ratio_sum * 12.0, 30.0)

        if not is_normal_folder:
            # Running from temp / hidden / untrusted paths (/tmp, /dev/shm, Downloads)
            # Automatic minimum floor of 45 (Suspicious)
            base_score = max(base_score, 45.0)

            # If untrusted path is coupled with network traffic, child shells, or high load -> Anomaly
            conn_val = row[FEATURES.index("connections")]
            child_val = row[FEATURES.index("children")]
            cpu_val = row[FEATURES.index("cpu_usage")]
            mem_val = row[FEATURES.index("memory_mb")]

            if conn_val >= 3 or child_val >= 2 or cpu_val > 40.0 or mem_val > 500.0 or ratio_sum > 1.0:
                base_score = max(base_score, 72.0)
        else:
            # Running from legitimate system folder (e.g. dev tools, compilers, scrapers)
            # Moderate spikes remain Suspicious (40 - 58) unless extreme (cpu > 85% or ratio_sum > 4.5)
            if base_score >= 60.0 and ratio_sum < 4.0 and row[FEATURES.index("cpu_usage")] < 85.0:
                base_score = 42.0 + min(15.0, ratio_sum * 4.0)

        scores.append(round(float(np.clip(base_score, 0.0, 100.0)), 1))

    return np.array(scores)


def classify_score(score: float) -> str:
    if score >= ANOMALY_SCORE:
        return "Anomaly"
    elif score >= SUSPICIOUS_SCORE:
        return "Suspicious"
    return "Benign"


def evaluate_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame, limits: dict):
    print(f"\n{'='*70}")
    print("STEP 5: Model Evaluation & Testing")
    print(f"{'='*70}")

    # 1. Baseline Validation (Check false-positive rate on normal training data)
    X_train = train_df[FEATURES].values.astype(float)
    train_scores = compute_risk_score(model, X_train, limits)
    fp_suspicious = int((train_scores >= SUSPICIOUS_SCORE).sum())
    fp_anomaly = int((train_scores >= ANOMALY_SCORE).sum())

    print("\n--- Baseline Check (Normal Training Data) ---")
    print(f"  Total samples tested        : {len(train_scores)}")
    print(f"  Mean baseline risk score    : {train_scores.mean():.1f}")
    print(f"  False positives (Suspicious): {fp_suspicious} ({fp_suspicious/len(train_scores)*100:.1f}%)")
    print(f"  False positives (Anomaly)   : {fp_anomaly} ({fp_anomaly/len(train_scores)*100:.1f}%)")

    # 2. Evaluation on Test Dataset (Mixed Threats)
    X_test = test_df[FEATURES].values.astype(float)
    test_scores = compute_risk_score(model, X_test, limits)
    test_labels = [classify_score(s) for s in test_scores]

    test_df_eval = test_df.copy()
    test_df_eval["risk_score"] = test_scores.round(1)
    test_df_eval["predicted_label"] = test_labels

    print("\n--- Test Set Evaluation Results (100 Samples) ---")
    expected = test_df["expected_label"].value_counts()
    predicted = pd.Series(test_labels).value_counts()

    print(f"  Expected Breakdown : {dict(expected)}")
    print(f"  Predicted Breakdown: {dict(predicted)}")

    # Confusion matrix
    conf_matrix = pd.crosstab(
        test_df["expected_label"],
        pd.Series(test_labels, name="Predicted"),
        margins=True
    )
    print("\nConfusion Matrix:")
    print(conf_matrix.to_string())

    # Detection Rates
    anomaly_mask = test_df["expected_label"] == "Anomaly"
    anomaly_detected = (pd.Series(test_labels)[anomaly_mask] == "Anomaly").sum()
    print(f"\nThreat Detection Accuracy:")
    print(f"  - Critical Anomaly Detection Rate : {anomaly_detected}/{anomaly_mask.sum()} ({anomaly_detected/anomaly_mask.sum()*100:.1f}%)")

    benign_mask = test_df["expected_label"] == "Benign"
    benign_correct = (pd.Series(test_labels)[benign_mask] == "Benign").sum()
    print(f"  - Benign Specificity (Normal Pass): {benign_correct}/{benign_mask.sum()} ({benign_correct/benign_mask.sum()*100:.1f}%)")

    # Save detailed prediction results
    test_df_eval.to_csv(PRED_FILE, index=False)
    print(f"\nDetailed test predictions saved to:\n  -> {PRED_FILE}")

    # Preview sample detections
    print("\nSample Detections Preview:")
    cols_to_show = ["process_name", "cpu_usage", "memory_mb", "connections", "normal_folder", "risk_score", "predicted_label", "expected_label"]
    print(test_df_eval[cols_to_show].sample(8, random_state=42).to_string(index=False))


# =============================================================================
# STEP 6 – Save Model Bundle
# =============================================================================
def save_model(model, limits: dict, medians: dict, train_count: int):
    print(f"\n{'='*70}")
    print("STEP 6: Saving Model Artifact")
    print(f"{'='*70}")

    bundle = {
        "model": model,
        "features": FEATURES,
        "limits": limits,
        "medians": medians,
        "contamination": CONTAMINATION,
        "thresholds": {
            "suspicious": SUSPICIOUS_SCORE,
            "anomaly": ANOMALY_SCORE
        },
        "train_samples": train_count
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Model saved successfully to:\n  -> {MODEL_FILE}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" OPTAD PROCESS THREAT DETECTION: MACHINE LEARNING PIPELINE")
    print("="*70)

    # 1. Load
    train_df, test_df = load_datasets()

    # 2. Clean
    train_df = clean_data(train_df, is_training=True)
    test_df  = clean_data(test_df,  is_training=False)

    # 3. EDA
    run_eda(train_df)

    # 4. Train
    model, limits, medians = train_model(train_df)

    # 5. Evaluate
    evaluate_model(model, train_df, test_df, limits)

    # 6. Save
    save_model(model, limits, medians, len(train_df))

    print(f"\n{'='*70}")
    print(" PIPELINE COMPLETE: MODEL READY FOR INFERENCE")
    print(f"{'='*70}\n")
