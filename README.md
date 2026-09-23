# MOT tracker benchmark

Nine trackers compared on Urban Tracker intersection video, all sharing one
cached set of YOLO26m detections. Output lands in `results/results.csv`.

## Setup

```bash
conda create -n mot python=3.10 && conda activate mot
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"   # must print True
pip install -r requirements.txt
```

Do not run `pip install boxmot`. Run the commands below instead, otherwise
every vehicle class scores 0.00.

```bash
git clone https://github.com/mikel-brostrom/boxmot ~/boxmot-repo
cd ~/boxmot-repo && git checkout v22.0.0
git apply ~/mot-pipeline/patches/cache_cls_column.patch
cp ~/mot-pipeline/patches/*.yaml boxmot/configs/benchmarks/
pip install -e .
```

## Prepare a sequence

Download annotations and frames from https://www.jpjodoin.com/urbantracker/
and unzip them. Build this layout, using Rouen as the example:

```
<root>/train_rouen/rouen/img1/000001.jpg ...
<root>/train_rouen/rouen/gt/gt.txt
<root>/train_rouen/rouen/det/det.txt
<root>/train_rouen/rouen/seqinfo.ini
```

`img1` holds only the annotated frame range, renumbered from 1. The offsets are
in `convert_urbantracker.py`. `seqinfo.ini` needs `name`, `imDir`, `frameRate`,
`seqLength`, `imWidth`, `imHeight`, `imExt`.

Give every benchmark its own split folder. boxmot scores every sequence it
finds under a split path, so two in one folder are pooled into one wrong score.

```bash
python convert_urbantracker.py --seq rouen \
    --sqlite <...>/rouen_gt.sqlite --out <root>/train_rouen/rouen/gt/gt.txt

python cache_detections.py yolo26m.pt \
    <root>/train_rouen/rouen/img1 <root>/train_rouen/rouen/det/det.txt
```

## Execution

Run from `~`. boxmot caches into `runs/` relative to the launch directory, so
running elsewhere builds a second, stale cache.

```bash
cd ~
python mot-pipeline/run_benchmark.py --data rouen
python mot-pipeline/extract_results.py --dataset rouen
```

Benchmarks are `stmarc`, `sherbrooke`, `rouen`, and `urbantracker` (the three
pooled). Add `--trackers bytetrack,botsort` for a subset.

Optional detector comparison, writing `results/detector_comparison.csv`:

```bash
python mot-pipeline/detector_eval.py
```

## Files

- `convert_urbantracker.py`: Polytrack sqlite annotations to MOT `gt.txt`
- `cache_detections.py`: run YOLO once, write MOT `det.txt`
- `run_benchmark.py`: run all nine trackers on one benchmark
- `extract_results.py`: summary JSONs to `results/results.csv`
- `detector_eval.py`: compare YOLO models by AP, recall and speed
- `patches/`: boxmot patch, benchmark configs, rebuild steps
- `results/`: output CSVs
