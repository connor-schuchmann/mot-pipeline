# added: converts Urban Tracker St-Marc Polytrack sqlite annotations to MOT gt.txt
#
# The St-Marc video is annotated from frame 1000 to 1999 (see the dataset ReadMe),
# so MOT frame numbering starts at 1 with an offset of 999.
#
# Six of the 28 tracks are typed "unknown" (2686 boxes, 32% of the dataset).
# They are recovered from the description field; box sizes and the per-class
# sqlite files (cars: 7 typed + 1 unknown, pedestrians: 13 typed + 5 unknown)
# both confirm the split below.
import argparse
import sqlite3
from pathlib import Path

# our gt class ids, referenced by evaluation.classes in stmarc.yaml
# bicycle is 9, not 3: boxmot remaps gt car 1 -> detector_id+1 = 3, which would
# collide with a bicycle id of 3 and score every cyclist as a car
CAR, PEDESTRIAN, BICYCLE = 1, 2, 9

# Polytrack road_user_type -> our gt class id
TYPE_MAP = {
    1: CAR,
    2: PEDESTRIAN,
    3: CAR,        # motorcycle, unused in St-Marc
    4: BICYCLE,
    5: CAR,        # bus, unused in St-Marc
    6: CAR,        # truck, unused in St-Marc
}

# road_user_type 0 ("unknown") resolved per object_id
UNKNOWN_MAP = {
    1: PEDESTRIAN,   # "black shirt"
    3: PEDESTRIAN,   # "waiting guy 1"
    10: PEDESTRIAN,  # "man in black"
    12: PEDESTRIAN,  # "white shirt, black hair, black pants"
    14: PEDESTRIAN,  # "fille chandail brun"
    23: CAR,         # "black acura"
}


def convert(sqlite_path: Path, out_path: Path, offset: int):
    con = sqlite3.connect(sqlite_path)

    types = dict(con.execute("select object_id, road_user_type from objects"))

    lines = []
    unresolved = set()
    frames = set()

    rows = con.execute(
        "select object_id, frame_number, x_top_left, y_top_left, "
        "x_bottom_right, y_bottom_right from bounding_boxes order by frame_number, object_id"
    )

    for object_id, frame_number, x1, y1, x2, y2 in rows:
        road_user_type = types.get(object_id, 0)
        if road_user_type == 0:
            cls_id = UNKNOWN_MAP.get(object_id)
        else:
            cls_id = TYPE_MAP.get(road_user_type)

        if cls_id is None:
            unresolved.add((object_id, road_user_type))
            continue

        mot_frame = frame_number - offset
        if mot_frame < 1:
            continue

        w = x2 - x1
        h = y2 - y1
        # MOT gt: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
        # track ids are 1-based, sqlite object_ids start at 0
        lines.append(
            f"{mot_frame},{object_id + 1},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,{cls_id},1"
        )
        frames.add(mot_frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Frames covered: {len(frames)} (MOT {min(frames)} to {max(frames)})")
    print(f"Annotation lines written: {len(lines)}")
    print(f"Output: {out_path}")
    if unresolved:
        print(f"WARNING: skipped unresolved objects (object_id, road_user_type): {unresolved}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=999,
                    help="video frame number of MOT frame 1, minus 1 (St-Marc: 999)")
    args = ap.parse_args()

    convert(Path(args.sqlite), Path(args.out), args.offset)
