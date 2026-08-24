"""
Benchmark runner: evaluates all boxmot-native trackers on the FastTracker
benchmark using cached YOLO26m detections, and writes one comparison CSV.

IMPORTANT: always run this from your home directory (~), in the boxmot conda env:
    cd ~ && python run_benchmark.py

Why: boxmot creates its cache folder ("runs/") inside whatever directory you
launch from. Launching from different directories creates separate, possibly
out-of-date caches (this caused silently-wrong results once already).
The one true cache lives at: ~/runs/dets_n_embs/fasttracker/
"""

import argparse
import csv
import json
import traceback
from datetime import datetime
from pathlib import Path

# out of box: boxmot config builder + eval engine
from boxmot.configs import build_mode_namespace, BOXMOT_DEFAULTS
from boxmot.engine.eval.evaluator import run_eval

# --- Configuration -----------------------------------------------------------

TRACKERS = [
    "bytetrack", "botsort", "strongsort", "ocsort", "deepocsort",
    "hybridsort", "boosttrack", "occluboost", "sfsort",
]

OUT_DIR = Path("/home/connor-schuchmann/benchmark_results")

# Base arguments matching your working CLI call:
# boxmot eval --benchmark fasttracker --detection-source public --split train

# --- Helpers -----------------------------------------------------------------

def flatten(d, prefix=""):
    """Flatten nested dicts: {"car": {"HOTA": 37}} -> {"car.HOTA": 37}."""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def main():
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    errors = {}

    for tracker in TRACKERS:
        print(f"\n=== Running {tracker} ===")
        # added: eval settings for each tracker
        payload = {
            "data": "fasttracker",
            "device": "0",
            "detector": [BOXMOT_DEFAULTS.shared.detector],
            "reid": [BOXMOT_DEFAULTS.shared.reid],
            "classes": None,
            "source": None,
            "benchmark": "",
            "split": "train",
            "detection_source": "public",
            "tune_kf": False,
            "tracker": tracker,
        }
        try:
            # out of box: runs tracking + metrics, same as the `boxmot eval` CLI
            args = build_mode_namespace("eval", payload, explicit_keys={"data", "split", "detection_source", "tracker", "device"})
            result = run_eval(args, verbose=False)

        except Exception:
            print(f"[SKIPPED] {tracker} failed:")
            traceback.print_exc()
            errors[tracker] = traceback.format_exc()
            continue

        # Save the full summary as JSON per tracker (raw record, debugging aid)
        with open(OUT_DIR / f"{stamp}_{tracker}_summary.json", "w") as f:
            json.dump(result.to_dict(include_raw=True), f, indent=2, default=str)

        # added: read per-class metrics from result.raw — result.summary is
        # unreliable for multi-class benchmarks (falls back to first class only)
        row = {"tracker": tracker}
        for cls_name, cls_metrics in result.raw.items():
            if not isinstance(cls_metrics, dict):
                continue
            row.update({
                f"{cls_name}.{key}": value
                for key, value in cls_metrics.items()
                if isinstance(value, (int, float))
            })
        rows.append(row)
        print(f"[OK] {tracker}")

    # added: merge all trackers into one comparison CSV
    if rows:
        # Union of all columns across trackers, "tracker" first
        columns = ["tracker"] + sorted({k for r in rows for k in r} - {"tracker"})
        csv_path = OUT_DIR / f"{stamp}_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote comparison CSV: {csv_path}")

    # --- Write error log if anything failed ---
    if errors:
        err_path = OUT_DIR / f"{stamp}_errors.log"
        with open(err_path, "w") as f:
            for tracker, tb in errors.items():
                f.write(f"===== {tracker} =====\n{tb}\n")
        print(f"{len(errors)} tracker(s) failed — details in {err_path}")

    print(f"\nDone: {len(rows)} succeeded, {len(errors)} failed.")


if __name__ == "__main__":
    main()