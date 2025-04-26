#!/usr/bin/env python3
"""
NAB → Prometheus exporter (fair parallel version)

Usage:
    python exporter.py --results_dir results --rate 30

Then browse:
    http://localhost:8000/metrics
Prometheus:
    scrape target: http://localhost:8000/metrics
"""

import argparse
import concurrent.futures
import pathlib
import threading
import time
from collections import defaultdict
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

# ---------------------------------------------------------------------------
# Prometheus registry + metric
# ---------------------------------------------------------------------------
reg = CollectorRegistry()
score_gauge = Gauge(
    "nab_anomaly_score",
    "Anomaly score per NAB file",
    labelnames=["model", "dataset", "file"],
    registry=reg,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()

@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def push(model: str, dataset: str, file_name: str, s: float) -> None:
    """Write one value into the Gauge (thread-safe)."""
    score_gauge.labels(model, dataset, file_name).set(s)

def detect_score_column(df: pd.DataFrame) -> str:
    """Return the first column whose name contains 'score' (case-insensitive)."""
    for col in df.columns:
        if "score" in col.lower():
            return col
    # fallback: third column if there was no header
    return df.columns[2]

def play_one(csv_path: pathlib.Path, rate_hz: float) -> None:
    """Stream a single CSV file."""
    try:
        model = csv_path.parents[1].name  # model folder
        dataset = csv_path.parent.name
        file_name = csv_path.name

        # Load CSV flexibly
        try:
            df = pd.read_csv(csv_path)
            score_col = detect_score_column(df)
        except pd.errors.ParserError:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["timestamp", "value", "anomaly_score", "label"],
            )
            score_col = "anomaly_score"

        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.dropna(subset=[score_col])
        if df.empty:
            print(f" Empty after cleaning: {csv_path}")
            return

        # Prime the series for Prometheus (first value)
        push(model, dataset, file_name, float(df[score_col].iloc[0]))

        # Stream the rest
        for s in df[score_col].iloc[1:]:
            push(model, dataset, file_name, float(s))
            time.sleep(1.0 / rate_hz)

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def start_replay_all(results_root: pathlib.Path, rate_hz: float, max_workers: Optional[int] = None) -> None:
    """Launch streaming threads for all matching CSVs, balanced across models."""
    csv_paths = [
        p
        for p in results_root.rglob("*.csv")
        if not p.name.endswith("_scores.csv") and not p.name.startswith("._")
    ]
    if not csv_paths:
        print(f"No CSV files found under {results_root}")
        return

    # --- group CSVs by model ---
    by_model = defaultdict(list)
    for path in csv_paths:
        model = path.parents[1].name
        by_model[model].append(path)

    # --- round-robin pick files across models ---
    queue = []
    while any(by_model.values()):
        for model in list(by_model.keys()):
            if by_model[model]:
                queue.append(by_model[model].pop(0))
            if not by_model[model]:
                del by_model[model]

    print(f"[exporter] Fair replay of {len(queue)} files across {len(set(p.parents[1].name for p in queue))} models")

    if max_workers is None:
        max_workers = min(len(queue), 32)

    print(f"[exporter] Using up to {max_workers} threads at {rate_hz} rows/sec per file")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for path in queue:
            pool.submit(play_one, path, rate_hz)

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stream NAB result CSVs to Prometheus")
    parser.add_argument("--results_dir", default="results", type=str,
                        help="Root directory that contains the NAB result folders (default=results/)")
    parser.add_argument("--rate", default=30.0, type=float,
                        help="Rows per second *per file*")
    parser.add_argument("--workers", default=None, type=int,
                        help="Max parallel files (default=min(num_csv,32))")
    parser.add_argument("--port", default=8000, type=int,
                        help="HTTP port for /metrics")
    args = parser.parse_args()

    results_root = pathlib.Path(args.results_dir).expanduser().resolve()

    print(f"[exporter] Streaming from: {results_root}")

    threading.Thread(
        target=start_replay_all,
        kwargs={
            "results_root": results_root,
            "rate_hz": args.rate,
            "max_workers": args.workers,
        },
        daemon=True,
    ).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
