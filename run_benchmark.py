"""Run every tracker on one benchmark and write a per-tracker summary JSON.

Run from ~: boxmot puts its cache in runs/ relative to the working directory,
so launching from elsewhere builds a second, stale cache.

    cd ~ && python mot-pipeline/run_benchmark.py --data stmarc
"""

import argparse
import csv
import json
import traceback
from datetime import datetime
from pathlib import Path

from boxmot.configs import build_mode_namespace, BOXMOT_DEFAULTS
from boxmot.engine.eval.evaluator import run_eval

TRACKERS = [
    "bytetrack", "botsort", "strongsort", "ocsort", "deepocsort",
    "hybridsort", "boosttrack", "occluboost", "sfsort",
]

OUT_DIR = Path.home() / "benchmark_results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="benchmark config name: stmarc, sherbrooke, rouen, urbantracker")
    ap.add_argument("--split", default="train")
    ap.add_argument("--trackers", default=None,
                    help="comma separated subset of trackers (default: all)")
    cli = ap.parse_args()

    trackers = cli.trackers.split(",") if cli.trackers else TRACKERS

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    errors = {}

    for tracker in trackers:
        print(f"\n=== Running {tracker} ===")
        payload = {
            "data": cli.data,
            "device": "0",
            "detector": [BOXMOT_DEFAULTS.shared.detector],
            "reid": [BOXMOT_DEFAULTS.shared.reid],
            "classes": None,
            "source": None,
            "benchmark": "",
            "split": cli.split,
            "detection_source": "public",
            "tune_kf": False,
            "tracker": tracker,
            "per_class": True,
            # one sequence at a time, else concurrent sequences inflate the
            # per-frame timings (accuracy metrics are unaffected)
            "n_threads": 1,
        }
        try:
            args = build_mode_namespace("eval", payload, explicit_keys={"data", "split", "detection_source", "tracker", "device", "per_class", "n_threads"})
            result = run_eval(args, verbose=False)

        except Exception:
            print(f"[SKIPPED] {tracker} failed:")
            traceback.print_exc()
            errors[tracker] = traceback.format_exc()
            continue

        with open(OUT_DIR / f"{stamp}_{cli.data}_{tracker}_summary.json", "w") as f:
            json.dump(result.to_dict(include_raw=True), f, indent=2, default=str)

        # read result.raw, not result.summary: summary reports the first class only
        row = {"tracker": tracker}
        for cls_name, cls_metrics in result.raw.items():
            if not isinstance(cls_metrics, dict):
                continue
            for key, value in cls_metrics.items():
                if isinstance(value, (int, float)):
                    row[f"{cls_name}.{key}"] = value
        rows.append(row)
        print(f"[OK] {tracker}")

    if rows:
        # every column any tracker produced, "tracker" first
        metric_names = set()
        for row in rows:
            for key in row:
                if key != "tracker":
                    metric_names.add(key)
        columns = ["tracker"] + sorted(metric_names)
        csv_path = OUT_DIR / f"{stamp}_{cli.data}_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote comparison CSV: {csv_path}")

    if errors:
        err_path = OUT_DIR / f"{stamp}_{cli.data}_errors.log"
        with open(err_path, "w") as f:
            for tracker, tb in errors.items():
                f.write(f"===== {tracker} =====\n{tb}\n")
        print(f"{len(errors)} tracker(s) failed, details in {err_path}")

    print(f"\nDone: {len(rows)} succeeded, {len(errors)} failed.")


if __name__ == "__main__":
    main()
