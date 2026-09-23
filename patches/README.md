# boxmot patches

Changes that live in the boxmot checkout, not in this repo, so a fresh clone
wipes them. Kept here so they survive a machine rebuild.

Tested against boxmot v22.0.0.

## Reapply after a fresh clone

    cd ~/boxmot-repo
    git checkout v22.0.0
    git apply ~/mot-pipeline/patches/cache_cls_column.patch
    cp ~/mot-pipeline/patches/*.yaml boxmot/configs/benchmarks/
    pip install -e .

Installing boxmot from PyPI instead drops the patch and every vehicle metric
scores 0.00.

## cache_cls_column.patch

`_generate_public_dets_cache` hardcoded the class column to zeros, so every
vehicle metric scored 0.00. Reads the class from MOT column 7 instead, falling
back to class 0 on negatives so MOT17-style det.txt still works.

## Benchmark configs

`stmarc`, `sherbrooke`, `rouen`, and `urbantracker` (the three pooled).

`detector_id` must be the COCO class id that lands in det.txt, not the gt class
id: boxmot writes tracker output as `cls + 1` and remaps gt to `detector_id + 1`.

Bicycle is gt class 9, not 3, because gt car 1 remaps to 3 and would collide,
scoring every cyclist as a car.

One benchmark per split directory. The metrics stage scores every sequence it
finds under the split path, so two sequences sharing a directory get silently
pooled into one score.

## GRAM

No converter here. An unrun draft is in git at `88e9ae5`, recoverable with
`git show 88e9ae5:convert_gram.py`. It was never validated against real data
and its class ids clash with the scheme above, so rewrite rather than trust it.
