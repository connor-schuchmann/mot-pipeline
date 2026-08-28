# boxmot patches

Changes that live in the boxmot checkout, not in this repo, so a fresh clone
wipes them. Kept here so they survive a machine rebuild.

Tested against boxmot v22.0.0.

## Reapply after a fresh clone

    cd ~/boxmot-repo
    git apply ~/mot-pipeline/patches/cache_cls_column.patch
    cp ~/mot-pipeline/patches/stmarc.yaml boxmot/configs/benchmarks/

## cache_cls_column.patch

`_generate_public_dets_cache` hardcoded the class column to zeros, so every
vehicle metric scored 0.00. Reads the class from MOT column 7 instead.

MOT17-style det.txt stores -1 in that column rather than a class id, so
negatives fall back to class 0 and pedestrian benchmarks keep working.

## stmarc.yaml

Benchmark config for the Urban Tracker St-Marc sequence.

`detector_id` must be the raw COCO class id that lands in det.txt, not the gt
class id. boxmot writes tracker output as `cls + 1` and remaps gt to
`detector_id + 1`, so the two meet in a shared numbering.

Unevaluated classes need ids well clear of that remap range. Bicycle is 9, not
3, because gt car 1 remaps to 3 and would collide with it, scoring every
cyclist as a car.
