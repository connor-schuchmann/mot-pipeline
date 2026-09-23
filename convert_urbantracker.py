"""Convert Urban Tracker Polytrack sqlite annotations to MOT gt.txt."""
import argparse
import sqlite3
from pathlib import Path

# bicycle is 9, not 3: boxmot remaps gt car 1 -> detector_id+1 = 3, which would
# collide with a bicycle id of 3 and score every cyclist as a car
CAR = 1
PEDESTRIAN = 2
BICYCLE = 9

# Polytrack road_user_type -> our gt class id
TYPE_MAP = {
    1: CAR,
    2: PEDESTRIAN,
    3: CAR,
    4: BICYCLE,
    5: CAR,
    6: CAR,
}

# offset = (first annotated video frame) - 1, from each sequence's ReadMe.
# "unknown" maps road_user_type 0 or -1 to a class, by sqlite description.
SEQUENCES = {
    "stmarc": {
        "offset": 999,
        "unknown": {
            1: PEDESTRIAN,   # "black shirt"
            3: PEDESTRIAN,   # "waiting guy 1"
            10: PEDESTRIAN,  # "man in black"
            12: PEDESTRIAN,  # "white shirt, black hair, black pants"
            14: PEDESTRIAN,  # "fille chandail brun"
            23: CAR,         # "black acura"
        },
    },
    "sherbrooke": {
        "offset": 2753,
        "unknown": {
            3: PEDESTRIAN,  # "Black Dress Women"
            5: CAR,         # "Black VUS"
        },
    },
    "rouen": {
        "offset": 19,
        "unknown": {
            5: PEDESTRIAN,  # "chandail noir"
            9: PEDESTRIAN,  # "enfant"
        },
    },
}


def convert(seq, sqlite_path, out_path):
    config = SEQUENCES[seq]
    offset = config["offset"]
    connection = sqlite3.connect(sqlite_path)

    type_of_object = {}
    for row in connection.execute("select object_id, road_user_type from objects"):
        object_id = row[0]
        road_user_type = row[1]
        type_of_object[object_id] = road_user_type

    lines = []
    unresolved = set()
    frames_seen = set()

    box_rows = connection.execute(
        "select object_id, frame_number, x_top_left, y_top_left, "
        "x_bottom_right, y_bottom_right from bounding_boxes order by frame_number, object_id"
    )

    for row in box_rows:
        object_id = row[0]
        frame_number = row[1]
        x1 = row[2]
        y1 = row[3]
        x2 = row[4]
        y2 = row[5]

        road_user_type = type_of_object.get(object_id, 0)
        class_id = TYPE_MAP.get(road_user_type)
        if class_id is None:
            class_id = config["unknown"].get(object_id)
        if class_id is None:
            unresolved.add((object_id, road_user_type))
            continue

        mot_frame = frame_number - offset
        if mot_frame < 1:
            continue

        # frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
        # track ids are 1-based, sqlite object_ids start at 0
        width = x2 - x1
        height = y2 - y1
        track_id = object_id + 1
        lines.append(
            f"{mot_frame},{track_id},{x1:.1f},{y1:.1f},{width:.1f},{height:.1f},1,{class_id},1"
        )
        frames_seen.add(mot_frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Frames covered: {len(frames_seen)} (MOT {min(frames_seen)} to {max(frames_seen)})")
    print(f"Annotation lines written: {len(lines)}")
    print(f"Output: {out_path}")
    if unresolved:
        print(f"WARNING: skipped unresolved objects (object_id, road_user_type): {unresolved}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, choices=sorted(SEQUENCES))
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    convert(args.seq, Path(args.sqlite), Path(args.out))
