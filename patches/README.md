# boxmot patches

Tested against boxmot v22.0.0.

## Reapply after a fresh clone

```bash
cd ~/boxmot-repo
git checkout v22.0.0
git apply ~/mot-pipeline/patches/cache_cls_column.patch
sed -i "s|/home/connor-schuchmann|$HOME|" ~/mot-pipeline/patches/*.yaml
cp ~/mot-pipeline/patches/*.yaml boxmot/configs/benchmarks/
pip install -e .
```

Installing boxmot from PyPI drops the patch and every vehicle metric scores 0.00.

## Files

- `cache_cls_column.patch`: makes boxmot read the detection class from MOT
  column 7, which it hardcoded to zero
- `stmarc.yaml`, `sherbrooke.yaml`, `rouen.yaml`, `urbantracker.yaml`:
  benchmark configs

## Notes

- `detector_id` is the COCO id in `det.txt`, not the gt id: boxmot writes
  tracker output as `cls + 1` and remaps gt to `detector_id + 1`
- one benchmark per split directory: the metrics stage scores every sequence
  under the split path and pools them into one wrong score
