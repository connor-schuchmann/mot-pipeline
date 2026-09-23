"""Pull the key metrics from run_benchmark.py's summary JSONs into one CSV.

    python ~/mot-pipeline/extract_results.py --dataset stmarc [--stamp 20260827_224430]
"""

import argparse
import csv
import json
from pathlib import Path

RESULTS_DIR = Path("/home/connor-schuchmann/benchmark_results")
OUT_CSV = RESULTS_DIR / "results.csv"

TRACKER_ORDER = [
    "bytetrack", "botsort", "strongsort", "ocsort", "deepocsort",
    "hybridsort", "boosttrack", "occluboost", "sfsort",
]

CLASS_COLUMNS = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]

COLUMNS = ["dataset", "detector", "tracker", "reid"]
for class_name in CLASS_COLUMNS:
    COLUMNS.append("HOTA_" + class_name)
COLUMNS = COLUMNS + [
    "HOTA_avg", "DetA_avg", "AssA_avg", "IDF1_avg", "MOTA_avg", "IDSW_total",
    "tracker_ms_per_frame", "tracker_fps", "n_frames", "n_gt_tracks",
]


def average(numbers):
    return sum(numbers) / len(numbers)


def is_run_stamp(text):
    """True for strings shaped like 20260827_224430."""
    if len(text) != 15 or text[8] != "_":
        return False

    date_part = text[0:8]
    time_part = text[9:15]
    return date_part.isdigit() and time_part.isdigit()


def find_summary_files(dataset):
    """Return {(stamp, tracker): path} for every matching summary JSON."""
    found = {}

    for path in RESULTS_DIR.glob("*_summary.json"):
        name = path.name

        stamp = name[0:15]
        if not is_run_stamp(stamp):
            continue

        middle = name.removesuffix("_summary.json")[16:]
        if not middle.startswith(dataset + "_"):
            continue

        tracker = middle[len(dataset) + 1:]
        found[(stamp, tracker)] = path

    return found


def extract_row(path, dataset, tracker):
    with open(path) as f:
        data = json.load(f)

    # read "raw", not "summary": summary reports the first class only
    classes = {}
    for class_name, metrics in data["raw"].items():
        if isinstance(metrics, dict):
            classes[class_name] = metrics
    if not classes:
        raise ValueError(path.name + ": no per-class metrics in 'raw'")

    hota_values = []
    deta_values = []
    assa_values = []
    idf1_values = []
    mota_values = []
    idsw_total = 0
    gt_tracks_total = 0

    for metrics in classes.values():
        hota_values.append(metrics["HOTA"])
        deta_values.append(metrics["DetA"])
        assa_values.append(metrics["AssA"])
        idf1_values.append(metrics["IDF1"])
        mota_values.append(metrics["MOTA"])
        idsw_total = idsw_total + metrics["IDSW"]
        gt_tracks_total = gt_tracks_total + metrics["GT_IDs"]

    exp_name = Path(data["exp_dir"]).name
    if "noreid" in exp_name:
        reid = "no"
    else:
        reid = "yes"

    timings = data["timings"]

    row = {
        "dataset": dataset,
        "detector": "YOLO26m",
        "tracker": tracker,
        "reid": reid,
        "HOTA_avg": round(average(hota_values), 2),
        "DetA_avg": round(average(deta_values), 2),
        "AssA_avg": round(average(assa_values), 2),
        "IDF1_avg": round(average(idf1_values), 2),
        "MOTA_avg": round(average(mota_values), 2),
        "IDSW_total": idsw_total,
        # association only: detections and reid embeddings come from cache
        "tracker_ms_per_frame": round(timings["avg_ms"]["track"], 2),
        "tracker_fps": round(timings["fps"], 1),
        "n_frames": timings["frames"],
        "n_gt_tracks": gt_tracks_total,
    }

    # blank, not 0, for a class this dataset does not evaluate; the averages
    # above likewise cover only the classes actually present
    for class_name in CLASS_COLUMNS:
        if class_name in classes:
            row["HOTA_" + class_name] = round(classes[class_name]["HOTA"], 2)
        else:
            row["HOTA_" + class_name] = ""

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="benchmark name, e.g. stmarc")
    ap.add_argument("--stamp", default=None,
                    help="run stamp YYYYMMDD_HHMMSS (default: newest for this dataset)")
    args = ap.parse_args()

    found = find_summary_files(args.dataset)
    if not found:
        raise SystemExit("No " + args.dataset + " summary JSONs in " + str(RESULTS_DIR))

    if args.stamp:
        stamp = args.stamp
    else:
        all_stamps = []
        for (found_stamp, _) in found:
            all_stamps.append(found_stamp)
        stamp = max(all_stamps)

    trackers = {}
    for (found_stamp, tracker), path in found.items():
        if found_stamp == stamp:
            trackers[tracker] = path
    if not trackers:
        raise SystemExit("No summaries for stamp " + stamp)
    print("Stamp " + stamp + ": " + str(len(trackers)) + " tracker(s)")

    ordered_names = []
    for tracker in TRACKER_ORDER:
        if tracker in trackers:
            ordered_names.append(tracker)
    for tracker in sorted(trackers):
        if tracker not in ordered_names:
            ordered_names.append(tracker)

    new_rows = []
    for tracker in ordered_names:
        new_rows.append(extract_row(trackers[tracker], args.dataset, tracker))

    # replace this dataset's rows, keep every other dataset's
    kept_rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            for old_row in csv.DictReader(f):
                if old_row["dataset"] != args.dataset:
                    kept_rows.append(old_row)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(kept_rows + new_rows)
    print("Wrote " + str(len(kept_rows) + len(new_rows)) + " rows to " + str(OUT_CSV))


if __name__ == "__main__":
    main()
