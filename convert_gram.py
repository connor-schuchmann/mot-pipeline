import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {
    "car": 1,
    "bus": 2,
    "motorbike": 3,
    "truck": 4,
    "van": 5,
    "big-truck": 6,
}


def convert(xml_dir: Path, out_path: Path, max_frames):
    xml_files = sorted(xml_dir.glob("*.xml"), key=lambda p: int(p.stem))

    lines = []
    unknown_classes = set()
    kept_frames = 0

    for xml_path in xml_files:
        frame_num = int(xml_path.stem)
        if max_frames is not None and frame_num >= max_frames:
            break

        mot_frame = frame_num + 1

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            cls_name = obj.findtext("class", default="").strip()
            track_id = obj.findtext("ID", default="").strip()
            bnd = obj.find("bndbox")
            if bnd is None or not track_id:
                continue

            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))

            w = xmax - xmin
            h = ymax - ymin

            cls_id = CLASS_MAP.get(cls_name)
            if cls_id is None:
                unknown_classes.add(cls_name)
                continue

            lines.append(
                f"{mot_frame},{int(track_id)},{xmin:.1f},{ymin:.1f},{w:.1f},{h:.1f},1,{cls_id},1"
            )
        kept_frames += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Frames processed: {kept_frames}")
    print(f"Annotation lines written: {len(lines)}")
    print(f"Output: {out_path}")
    if unknown_classes:
        print(f"WARNING: skipped unknown classes: {unknown_classes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    convert(Path(args.xml_dir), Path(args.out), args.max_frames)
