# MOT tracker benchmark on urban intersection video

Nine multi-object trackers compared on Urban Tracker traffic footage, toward
choosing a tracker for a college-campus intersection deployment with cars,
pedestrians, cyclists and micromobility.

Every tracker sees **identical detections**: YOLO26m is run once per sequence
and cached to `det.txt`, so differences between trackers are differences in
association, not in detection luck.

## Results

Full numbers in [`results/results.csv`](results/results.csv), 36 rows: nine
trackers on each of three sequences plus a pooled row.

Pooled across all three sequences (2602 frames, 64 ground-truth tracks):

| tracker | ReID | HOTA person | HOTA car | HOTA bicycle | HOTA avg | ID switches | ms/frame |
|---|---|---|---|---|---|---|---|
| botsort | yes | 43.26 | 31.15 | 9.56 | **27.99** | 19 | 4.45 |
| bytetrack | no | **45.96** | 27.16 | 9.40 | 27.51 | **17** | 2.33 |
| hybridsort | yes | 42.99 | 26.51 | **11.90** | 27.13 | 73 | 10.82 |
| strongsort | yes | 34.67 | **32.49** | 5.78 | 24.31 | 39 | 10.05 |
| occluboost | yes | 41.31 | 23.98 | 6.96 | 24.08 | 24 | 4.83 |
| deepocsort | yes | 35.19 | 30.23 | 5.61 | 23.68 | 40 | 5.21 |
| sfsort | no | 29.48 | 32.72 | 5.11 | 22.44 | 131 | **0.38** |
| ocsort | no | 29.81 | 31.41 | 3.60 | 21.61 | 19 | 2.20 |
| boosttrack | yes | 35.74 | 22.21 | 5.89 | 21.28 | 24 | 5.24 |

**BotSORT and ByteTrack lead**, and are the only trackers in the top three on
every individual sequence as well. ByteTrack does it with no ReID model and at
half the cost, which makes it the better default; BotSORT buys a little car
accuracy for roughly twice the compute.

**Cyclists are the weak point for every tracker.** COCO's `bicycle` class
covers the bike alone while the ground truth boxes wrap rider and bike
together, so only about a third of ground-truth bicycle boxes match a bicycle
detection at IoU 0.5. Detector choice does not fix this: YOLOv5, v8, v11 and
v8x all show the same person/bicycle label flicker (see
[`results/detector_comparison.csv`](results/detector_comparison.csv), where
YOLO26m is nonetheless the strongest of the four on bicycles). A stable
cyclist class would need fine-tuning on a rider-inclusive label.

Two caveats when reading the numbers. `HOTA_avg` averages only the classes a
sequence actually has, so it is not comparable across sequences with different
class counts. And `tracker_ms_per_frame` measures association only, since
detections and ReID embeddings are served from cache; end-to-end throughput is
dominated by the detector at roughly 25 ms/frame on an RTX 3060.

## Benchmarks

| name | sequence | frames | GT tracks (person / car / bicycle) |
|---|---|---|---|
| `stmarc` | Urban Tracker St-Marc | 1000 | 18 / 8 / 2 |
| `sherbrooke` | Urban Tracker Sherbrooke | 1001 | 5 / 15 / 0 |
| `rouen` | Urban Tracker Rouen | 601 | 11 / 4 / 1 |
| `urbantracker` | all three pooled | 2602 | 34 / 27 / 3 |

`urbantracker` pools the raw counts and recomputes the metrics, which is not
the same as averaging the three scores, because HOTA is not linear in those
counts. Both views are in the CSV.

## Reproducing

Requires the patched boxmot checkout described in
[`patches/README.md`](patches/README.md); a stock `pip install boxmot` scores
0.00 on every vehicle class.

```bash
python convert_urbantracker.py --seq stmarc --sqlite <stmarc_gt.sqlite> --out <.../gt/gt.txt>
python cache_detections.py yolo26m.pt <.../img1> <.../det/det.txt>

cd ~ && python mot-pipeline/run_benchmark.py --data stmarc   # all 9 trackers
python mot-pipeline/extract_results.py --dataset stmarc      # append to results.csv
```

Run `run_benchmark.py` from `~`: boxmot puts its cache in `runs/` relative to
the working directory, so launching elsewhere silently builds a second one.

## Files

| file | purpose |
|---|---|
| `convert_urbantracker.py` | Polytrack sqlite annotations to MOT `gt.txt` |
| `cache_detections.py` | run YOLO once, write MOT `det.txt` |
| `run_benchmark.py` | run all nine trackers on one benchmark |
| `extract_results.py` | summary JSONs to `results/results.csv` |
| `detector_eval.py` | compare YOLO models: per-class AP, recall, speed |
| `patches/` | boxmot patch and benchmark configs, with rebuild steps |
