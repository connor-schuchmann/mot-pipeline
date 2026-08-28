from ultralytics import YOLO
import time
import sys

model_name = sys.argv[1]   # e.g. yolo26m.pt
video = sys.argv[2]
output = sys.argv[3]

# out of box: ultralytics YOLO COCO detector (downloads weights on first use)
model = YOLO(model_name)
keep = {0,1,2,3,5,7} # {0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck}

start = time.time()

with open(output, "w") as f:
    # out of box: YOLO video inference
    # stream=True: uses a single frame's information and discards it instead of holding all frames in RAM
    # verbose=False: prevents a new print for each frame
    for frame, result in enumerate(model.predict(video, conf=0.1, stream=True, verbose=False), start=1):
        # added: filter classes and write detections as MOT-format det.txt
        for box in result.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cls = int(box.cls)
            if cls not in keep:
                continue # skip if irrelevant class
            conf = float(box.conf)
            w = x2 - x1
            h = y2 - y1
            # MOT: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, x, y, z
            f.write(f"{frame},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls},-1,-1\n")

elapsed = time.time() - start
print(f"{elapsed:.1f}s, {frame} frames, {frame/elapsed:.1f} FPS")