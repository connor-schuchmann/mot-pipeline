# MOT tracker benchmark

Compares nine multi-object trackers (ByteTrack, BotSORT, StrongSORT, OC-SORT,
DeepOC-SORT, HybridSORT, BoostTrack, OccluBoost, SFSORT) on Urban Tracker
intersection video.

YOLO26m runs once per sequence and its detections are cached to `det.txt`, so
every tracker sees identical input and the differences are in association, not
detection. Results are written to `results/results.csv`.

## 1. Environment

```bash
conda create -n mot python=3.10
conda activate mot

# install a CUDA torch build first; plain pip picks a CPU one
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"   # must print True

pip install -r requirements.txt
```

## 2. Patched boxmot

The pipeline needs a one-line fix in boxmot. Without it every vehicle class
scores 0.00.

```bash
git clone https://github.com/mikel-brostrom/boxmot ~/boxmot-repo
cd ~/boxmot-repo
git checkout v22.0.0
git apply ~/mot-pipeline/patches/cache_cls_column.patch
cp ~/mot-pipeline/patches/*.yaml boxmot/configs/benchmarks/
pip install -e .
```

See `patches/README.md` for what the patch does.

## 3. Get a sequence

Download annotations and frames from https://www.jpjodoin.com/urbantracker/
and unzip them. Sherbrooke and Rouen ship a `_frames.zip`; St-Marc ships only
`stmarc_video.avi`, so extract its frames yourself with ffmpeg.

## 4. Build the MOT folder

Each sequence needs this layout, where `<root>` is the benchmark root named in
its yaml under `patches/`:

```
<root>/<split>/<seq>/img1/000001.jpg ...   frames, renumbered from 1
<root>/<split>/<seq>/gt/gt.txt             ground truth
<root>/<split>/<seq>/det/det.txt           cached detections
<root>/<split>/<seq>/seqinfo.ini           name, imDir, frameRate, seqLength, imWidth, imHeight, imExt
```

Only part of the annotated video is labelled, so copy just that frame range
into `img1/` and renumber it to start at 1. Rouen for example is annotated over
video frames 20 to 620, which become MOT frames 1 to 601. The per-sequence
offsets are listed in `convert_urbantracker.py`.

Each benchmark needs its **own split folder**. boxmot scores every sequence it
finds under a split path, so two sequences sharing one folder are silently
pooled into a single wrong score.

Then generate the ground truth and the detections:

```bash
python convert_urbantracker.py --seq rouen \
    --sqlite <...>/rouen_gt.sqlite --out <root>/train_rouen/rouen/gt/gt.txt

python cache_detections.py yolo26m.pt \
    <root>/train_rouen/rouen/img1 <root>/train_rouen/rouen/det/det.txt
```

## 5. Run the benchmark

Run from your home directory. boxmot creates its cache in `runs/` relative to
wherever you launch from, so running elsewhere silently builds a second one.

```bash
cd ~
python mot-pipeline/run_benchmark.py --data rouen        # all nine trackers
python mot-pipeline/extract_results.py --dataset rouen   # adds rows to results.csv
```

Benchmark names are `stmarc`, `sherbrooke`, `rouen`, and `urbantracker` (all
three pooled). Add `--trackers bytetrack,botsort` to run a subset.

`extract_results.py` replaces only the rows for the dataset you name, so other
datasets already in the CSV are left alone.

## Optional: compare detectors

```bash
python mot-pipeline/detector_eval.py
```

Runs YOLOv5m, YOLOv8m and YOLO26m over the converted sequences and writes
per-class AP, recall and speed to `results/detector_comparison.csv`. It reads
`img1/` and `gt.txt` only, and does not touch the cached detections.

## Files

- `convert_urbantracker.py`: Urban Tracker Polytrack sqlite annotations to MOT `gt.txt`
- `cache_detections.py`: run YOLO once over a sequence, write MOT `det.txt`
- `run_benchmark.py`: run all nine trackers on one benchmark
- `extract_results.py`: per-tracker summary JSONs to `results/results.csv`
- `detector_eval.py`: compare YOLO models by per-class AP, recall and speed
- `patches/`: the boxmot patch, the four benchmark configs, and rebuild steps
- `results/`: the output CSVs

## Reading the numbers

`HOTA_avg` averages only the classes a sequence actually contains, so it is not
comparable between sequences with different class counts. Blank class columns
mean the ground truth has none of that class, not a score of zero.

`tracker_ms_per_frame` is association time only, since detections and ReID
embeddings come from cache. End-to-end speed is dominated by the detector, at
roughly 25 ms/frame on an RTX 3060.
