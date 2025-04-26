import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

# === CONFIG ===
MODEL_NAME = "my_model"
NAB_ROOT = Path("NAB")
DATA_DIR = NAB_ROOT / "data"
RESULTS_DIR = NAB_ROOT / "results" / MODEL_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def process_file(file_path: Path, output_path: Path):
    # === 1. Load & clean data ===
    df = pd.read_csv(file_path, header=None, names=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["timestamp", "value"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # === 2. Feature engineering ===
    df["delta"] = df["value"].diff().abs()
    df["rolling_mean"] = df["value"].rolling(window=12, min_periods=1).mean()
    df["rolling_std"] = df["value"].rolling(window=12, min_periods=1).std()
    df["z_score"] = (df["value"] - df["rolling_mean"]) / (df["rolling_std"] + 1e-6)
    df["time_numeric"] = df["timestamp"].astype(np.int64) // 10**9

    # === 3. Prepare features ===
    features = ["value", "delta", "rolling_mean", "rolling_std", "z_score", "time_numeric"]
    df_features = df[features].fillna(0)

    # === 4. Isolation Forest ===
    model = IsolationForest(contamination=0.002, random_state=42)
    df["raw_score"] = model.fit_predict(df_features)
    df["raw_score"] = (df["raw_score"] == -1).astype(int)  # 1 = anomaly

    # === 5. Smoothing (3-point majority vote) ===
    df["smoothed_score"] = df["raw_score"].rolling(3, min_periods=1).mean()
    df["anomaly_score"] = (df["smoothed_score"] > 0.5).astype(int)

    # === 6. Output in NAB format ===
    output_df = df[["timestamp", "value", "anomaly_score"]].copy()
    output_df["label"] = ""  # NAB uses combined_windows.json for real labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

def main():
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            for file_path in dataset_dir.glob("*.csv"):
                dataset = dataset_dir.name
                output_filename = f"{MODEL_NAME}_{file_path.name}"
                output_path = RESULTS_DIR / dataset / output_filename
                print(f"🌀 Processing: {file_path.name} → {output_path}")
                process_file(file_path, output_path)

if __name__ == "__main__":
    main()
