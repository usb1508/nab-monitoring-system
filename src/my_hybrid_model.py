import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
import ruptures as rpt
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
MODEL_NAME = "my_model_1"
NAB_ROOT = Path("NAB")
DATA_DIR = NAB_ROOT / "data"
RESULTS_DIR = NAB_ROOT / "results" / MODEL_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def extract_features(df):
    df["delta"] = df["value"].diff().abs()
    df["rolling_mean"] = df["value"].rolling(12, min_periods=1).mean()
    df["rolling_std"] = df["value"].rolling(12, min_periods=1).std()
    df["z_score"] = (df["value"] - df["rolling_mean"]) / (df["rolling_std"] + 1e-6)
    df["time_numeric"] = df["timestamp"].astype(np.int64) // 10**9
    return df.fillna(0)

def stl_decompose(df):
    try:
        stl = STL(df["value"], period=12)
        res = stl.fit()
        df["resid"] = res.resid
    except:
        df["resid"] = 0
    return df

def changepoint_score(series):
    try:
        algo = rpt.Pelt(model="rbf").fit(series)
        bkpts = algo.predict(pen=10)
        cp_mask = np.zeros(len(series))
        for bp in bkpts[:-1]:
            cp_mask[bp] = 1
        return cp_mask
    except:
        return np.zeros(len(series))

def detect_anomalies(df):
    features = ["value", "delta", "rolling_mean", "rolling_std", "z_score", "time_numeric", "resid"]
    X = df[features].fillna(0)

    clf = IsolationForest(contamination=0.005, random_state=42)
    df["iforest_score"] = clf.fit_predict(X)
    df["iforest_score"] = (df["iforest_score"] == -1).astype(float)

    df["changepoint"] = changepoint_score(df["value"].values)

    df["ensemble_score"] = (
        0.6 * df["iforest_score"] +
        0.3 * (np.abs(df["z_score"]) > 2).astype(float) +
        0.1 * df["changepoint"]
    )

    return df

def process_file(file_path, output_path):
    try:
        df = pd.read_csv(file_path, header=None, names=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(subset=["timestamp", "value"], inplace=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        df = extract_features(df)
        df = stl_decompose(df)
        df = detect_anomalies(df)

        out_df = df[["timestamp", "value", "ensemble_score"]].copy()
        out_df.columns = ["timestamp", "value", "anomaly_score"]
        out_df["label"] = ""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        print(f"{file_path.name} processed.")
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

def main():
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for dataset_dir in DATA_DIR.iterdir():
            if dataset_dir.is_dir():
                for file_path in dataset_dir.glob("*.csv"):
                    dataset = dataset_dir.name
                    output_filename = f"{MODEL_NAME}_{file_path.name}"
                    output_path = RESULTS_DIR / dataset / output_filename
                    tasks.append(executor.submit(process_file, file_path, output_path))

        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print("Thread error:", e)

if __name__ == "__main__":
    main()
