from ultralytics import YOLO
from pathlib import Path
import time
import sys

model_name = sys.argv[1]   # e.g. yolo26m.pt
video = sys.argv[2]
output = sys.argv[3]

model = YOLO(model_name)
# wider than the benchmark configs, which map only 0/1/2; the rest are written
# to det.txt but no eval class claims them
keep = {0,1,2,3,5,7} # {0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck}

start = time.time()
frame = 0

# the det/ folder may not exist yet
Path(output).parent.mkdir(parents=True, exist_ok=True)

with open(output, "w") as f:
    # stream=True: hold one frame at a time instead of all of them in RAM
    for frame, result in enumerate(model.predict(video, conf=0.1, stream=True, verbose=False), start=1):
        for box in result.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cls = int(box.cls)
            if cls not in keep:
                continue
            conf = float(box.conf)
            w = x2 - x1
            h = y2 - y1
            # frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, x, y, z
            # column 8 (class) is non-standard; boxmot is patched to read it
            f.write(f"{frame},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls},-1,-1\n")

elapsed = time.time() - start
if frame == 0:
    print(f"No frames read from {video}")
else:
    print(f"{elapsed:.1f}s, {frame} frames, {frame/max(elapsed, 1e-9):.1f} FPS")
