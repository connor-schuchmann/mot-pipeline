"""Compare YOLO detectors on the converted Urban Tracker datasets.

Per model and dataset: per-class AP50-95 and AP50 (COCO protocol), recall at
conf 0.1 / IoU 0.5, and RTX 3060 / CPU speeds. Reads img1/ and gt.txt only.

AP here is comparative, not absolute: the gt annotates moving road users, so a
correct parked-car detection scores as a false positive for every model alike.

    python ~/mot-pipeline/detector_eval.py
"""

import csv
import time
from pathlib import Path

MODELS = ["yolov5mu.pt", "yolov8m.pt", "yolo26m.pt"]

ROOT = Path.home() / "UrbanTracker/datasets/UrbanTracker-Benchmark"
DATASETS = {
    "stmarc": ROOT / "train/stmarc",
    "sherbrooke": ROOT / "train_sherbrooke/sherbrooke",
    "rouen": ROOT / "train_rouen/rouen",
}

# gt class id -> name; detector COCO id -> gt class id
GT_CLASSES = {1: "car", 2: "person", 9: "bicycle"}
DET_TO_GT = {2: 1, 0: 2, 1: 9}

OUT_CSV = Path.home() / "benchmark_results/detector_comparison.csv"
RECALL_CONF = 0.1
RECALL_IOU = 0.5
MAX_DETS_PER_FRAME = 100

# COCO's ten IoU thresholds: 0.50, 0.55, ... 0.95
IOU_THRESHOLDS = []
for step in range(10):
    IOU_THRESHOLDS.append(0.5 + 0.05 * step)


def iou(box_a, box_b):
    """Overlap ratio of two [x1, y1, x2, y2] boxes."""
    overlap_width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    overlap_height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0

    intersection = overlap_width * overlap_height
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def load_gt(seq_dir):
    """gt.txt -> {class_id: {frame: [box, box, ...]}}"""
    gt_boxes = {}

    for line in open(seq_dir / "gt/gt.txt"):
        fields = line.split(",")
        frame = int(fields[0])
        class_id = int(fields[7])

        x = float(fields[2])
        y = float(fields[3])
        w = float(fields[4])
        h = float(fields[5])
        box = [x, y, x + w, y + h]

        if class_id not in gt_boxes:
            gt_boxes[class_id] = {}
        if frame not in gt_boxes[class_id]:
            gt_boxes[class_id][frame] = []
        gt_boxes[class_id][frame].append(box)

    return gt_boxes


def collect_detections(model, seq_dir, needed_classes):
    """Run the model over img1/ once -> {class_id: {frame: [(conf, box), ...]}}

    conf 0.001 because AP needs the full confidence sweep; the recall column
    re-filters these at 0.1 rather than running inference twice. Detections of
    a class this dataset has no gt for are dropped, since nothing scores them.
    """
    detections = {}
    for class_id in needed_classes:
        detections[class_id] = {}

    results = model.predict(str(seq_dir / "img1"), conf=0.001, stream=True,
                            verbose=False, device=0, classes=list(DET_TO_GT))

    frame = 0
    for result in results:
        frame = frame + 1

        for box in result.boxes:
            class_id = DET_TO_GT[int(box.cls)]
            if class_id not in needed_classes:
                continue
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()

            if frame not in detections[class_id]:
                detections[class_id][frame] = []
            detections[class_id][frame].append((conf, xyxy))

    return detections


def prepare_frames(dets_by_frame, gt_by_frame):
    """Per frame: confidences plus an IoU table, reused by all ten thresholds."""
    prepared = []

    for frame, dets in dets_by_frame.items():
        # tuples compare element by element, so (conf, box) pairs sort by conf
        dets = sorted(dets, reverse=True)
        dets = dets[:MAX_DETS_PER_FRAME]
        gts = gt_by_frame.get(frame, [])

        confidences = []
        iou_table = []  # iou_table[i][j] = IoU of detection i vs gt box j
        for conf, det_box in dets:
            confidences.append(conf)

            row = []
            for gt_box in gts:
                row.append(iou(det_box, gt_box))
            iou_table.append(row)

        prepared.append({"confidences": confidences, "iou_table": iou_table,
                         "n_gt": len(gts)})

    return prepared


def match_at_threshold(prepared_frames, iou_threshold):
    """COCO greedy matching: most confident detection first, each gt box claimed
    once, so a duplicate detection of the same object is a false positive."""
    scored = []

    for frame in prepared_frames:
        gt_claimed = [False] * frame["n_gt"]

        for i in range(len(frame["confidences"])):
            best_j = -1
            best_iou = 0.0
            for j in range(frame["n_gt"]):
                if gt_claimed[j]:
                    continue
                if frame["iou_table"][i][j] >= iou_threshold:
                    if frame["iou_table"][i][j] > best_iou:
                        best_iou = frame["iou_table"][i][j]
                        best_j = j

            if best_j >= 0:
                gt_claimed[best_j] = True
                scored.append((frame["confidences"][i], True))
            else:
                scored.append((frame["confidences"][i], False))

    scored.sort(reverse=True)
    return scored


def average_precision(scored, n_gt):
    """101-point interpolated AP."""
    if n_gt == 0 or len(scored) == 0:
        return 0.0

    # running precision and recall down the ranked list
    recalls = []
    precisions = []
    true_positives = 0
    seen = 0
    for _, was_tp in scored:
        seen = seen + 1
        if was_tp:
            true_positives = true_positives + 1
        recalls.append(true_positives / n_gt)
        precisions.append(true_positives / seen)

    # envelope: best precision from each point onward
    envelope = [0.0] * len(precisions)
    best = 0.0
    for i in range(len(precisions) - 1, -1, -1):
        if precisions[i] > best:
            best = precisions[i]
        envelope[i] = best

    # sample the envelope at recall 0.00, 0.01, ... 1.00
    total = 0.0
    pointer = 0
    for level_index in range(101):
        level = level_index / 100
        while pointer < len(recalls) and recalls[pointer] < level:
            pointer = pointer + 1
        if pointer < len(recalls):
            total = total + envelope[pointer]

    return total / 101


def eval_model_on_dataset(model, seq_dir):
    gt_boxes = load_gt(seq_dir)
    detections = collect_detections(model, seq_dir, set(gt_boxes))

    row = {}
    ap5095_values = []
    ap50_values = []

    for class_id, class_name in GT_CLASSES.items():
        # blank, not 0, for a class absent from this dataset's gt
        if class_id not in gt_boxes:
            row["AP50_95_" + class_name] = ""
            row["AP50_" + class_name] = ""
            row["recall_" + class_name] = ""
            continue

        gt_for_class = gt_boxes[class_id]
        n_gt = 0
        for frame_boxes in gt_for_class.values():
            n_gt = n_gt + len(frame_boxes)

        prepared = prepare_frames(detections[class_id], gt_for_class)
        ap_per_threshold = []
        for threshold in IOU_THRESHOLDS:
            scored = match_at_threshold(prepared, threshold)
            ap_per_threshold.append(average_precision(scored, n_gt))
        ap5095 = sum(ap_per_threshold) / len(ap_per_threshold)
        ap50 = ap_per_threshold[0]

        # recall at the operating point the tracker consumes detections at
        strong_dets = {}
        for frame, dets in detections[class_id].items():
            kept = []
            for conf, box in dets:
                if conf >= RECALL_CONF:
                    kept.append((conf, box))
            strong_dets[frame] = kept

        strong_prepared = prepare_frames(strong_dets, gt_for_class)
        scored = match_at_threshold(strong_prepared, RECALL_IOU)
        found = 0
        for _, was_tp in scored:
            if was_tp:
                found = found + 1

        row["AP50_95_" + class_name] = round(100 * ap5095, 1)
        row["AP50_" + class_name] = round(100 * ap50, 1)
        row["recall_" + class_name] = round(100 * found / n_gt, 1)
        ap5095_values.append(ap5095)
        ap50_values.append(ap50)

    row["mAP50_95"] = round(100 * sum(ap5095_values) / len(ap5095_values), 1)
    row["mAP50"] = round(100 * sum(ap50_values) / len(ap50_values), 1)
    return row


def measure_speed(model):
    """predict() timings including pre/postprocessing, on stmarc frames."""
    paths = sorted((DATASETS["stmarc"] / "img1").glob("*.jpg"))
    timings = {}

    # GPU, one image at a time
    for p in paths[:10]:
        model.predict(str(p), device=0, verbose=False)
    start = time.perf_counter()
    for p in paths[10:210]:
        model.predict(str(p), device=0, verbose=False)
    timings["gpu_b1_ms"] = round((time.perf_counter() - start) / 200 * 1000, 1)

    # GPU, batches of 32
    batch = []
    for p in paths[:320]:
        batch.append(str(p))
    model.predict(batch[:32], device=0, verbose=False)
    start = time.perf_counter()
    for i in range(32, 320, 32):
        model.predict(batch[i:i + 32], device=0, verbose=False)
    timings["gpu_b32_ms"] = round((time.perf_counter() - start) / 288 * 1000, 2)

    # CPU, one image at a time; 30 frames is enough, it is slow
    model.predict(str(paths[0]), device="cpu", verbose=False)
    start = time.perf_counter()
    for p in paths[1:31]:
        model.predict(str(p), device="cpu", verbose=False)
    timings["cpu_b1_ms"] = round((time.perf_counter() - start) / 30 * 1000, 0)

    return timings


def main():
    from ultralytics import YOLO

    columns = ["model", "dataset", "mAP50_95", "mAP50"]
    for class_name in GT_CLASSES.values():
        columns.append("AP50_95_" + class_name)
    for class_name in GT_CLASSES.values():
        columns.append("AP50_" + class_name)
    for class_name in GT_CLASSES.values():
        columns.append("recall_" + class_name)
    columns = columns + ["gpu_b1_ms", "gpu_b32_ms", "cpu_b1_ms"]

    rows = []
    for model_path in MODELS:
        name = Path(model_path).stem
        model = YOLO(model_path)

        print("=== " + name + ": speed ===", flush=True)
        speed = measure_speed(model)

        for ds_name, seq_dir in DATASETS.items():
            print("=== " + name + " on " + ds_name + " ===", flush=True)
            row = {"model": name, "dataset": ds_name}
            row.update(speed)
            row.update(eval_model_on_dataset(model, seq_dir))
            rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote " + str(len(rows)) + " rows to " + str(OUT_CSV))


if __name__ == "__main__":
    main()
