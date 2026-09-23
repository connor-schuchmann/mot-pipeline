# MOT tracker benchmark

Nine trackers compared on Urban Tracker intersection video, all sharing one
cached set of YOLO26m detections. Output lands in `~/benchmark_results/results.csv`.

## Setup

```bash
git clone https://github.com/connor-schuchmann/mot-pipeline ~/mot-pipeline
cd ~/mot-pipeline

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
sed -i "s|/home/connor-schuchmann|$HOME|" ~/mot-pipeline/patches/*.yaml
cp ~/mot-pipeline/patches/*.yaml boxmot/configs/benchmarks/
pip install -e .
```

## Prepare a sequence

Download from https://www.jpjodoin.com/urbantracker/ and unzip each one into
`~/UrbanTracker/<name>/`:

1. Rouen: `rouen_annotations.zip` and `rouen_frames.zip`
2. Sherbrooke: `sherbrooke_annotations.zip` and `sherbrooke_frames.zip`
3. St-Marc: `stmarc_annotations.zip` and `stmarc_video.avi`

St-Marc ships no frames zip, so build its frames folder from the video:

```bash
mkdir -p ~/UrbanTracker/stmarc/stmarc_frames
ffmpeg -i ~/UrbanTracker/stmarc/stmarc_video.avi ~/UrbanTracker/stmarc/stmarc_frames/%08d.jpg
```

Then run this block three times, once per sequence:

```bash
cd ~/mot-pipeline
SEQ=rouen                                                          # then sherbrooke, then stmarc
DEST=~/UrbanTracker/datasets/UrbanTracker-Benchmark/train_$SEQ/$SEQ   # one folder per sequence

# writes gt.txt, img1/ and seqinfo.ini
python convert_urbantracker.py --seq $SEQ --sqlite ~/UrbanTracker/$SEQ/${SEQ}_annotations/${SEQ}_gt.sqlite --out $DEST/gt/gt.txt --frames ~/UrbanTracker/$SEQ/${SEQ}_frames

# writes det.txt
python cache_detections.py yolo26m.pt $DEST/img1 $DEST/det/det.txt
```

Do not put two sequences in one `train_` folder because boxmot scores everything together 
and gives wrong score.

To get the pooled `urbantracker` benchmark link the three together:

```bash
cd ~/UrbanTracker/datasets/UrbanTracker-Benchmark
mkdir -p train_urbantracker
ln -sfn $PWD/train_stmarc/stmarc train_urbantracker/stmarc
ln -sfn $PWD/train_sherbrooke/sherbrooke train_urbantracker/sherbrooke
ln -sfn $PWD/train_rouen/rouen train_urbantracker/rouen
```

## Execution

```bash
cd ~
python mot-pipeline/run_benchmark.py --data rouen
python mot-pipeline/extract_results.py --dataset rouen
```

Benchmarks are `stmarc`, `sherbrooke`, `rouen`, and `urbantracker` (the three
pooled). Add `--trackers bytetrack,botsort` for a subset.

Optional detector comparison, writing `~/benchmark_results/detector_comparison.csv`:

```bash
python mot-pipeline/detector_eval.py
```

## Files

- `convert_urbantracker.py`: Polytrack sqlite annotations to MOT `gt.txt`
- `cache_detections.py`: run YOLO once, write MOT `det.txt`
- `run_benchmark.py`: run all nine trackers on one benchmark
- `extract_results.py`: summary JSONs to `~/benchmark_results/results.csv`
- `detector_eval.py`: compare YOLO models by AP, recall and speed
- `patches/`: boxmot patch, benchmark configs, rebuild steps
- `results/`: published CSVs from our runs, for reference; your own go to `~/benchmark_results/`
