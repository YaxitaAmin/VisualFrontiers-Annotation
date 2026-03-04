#!/usr/bin/env python3

import csv
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

# Directory containing action annotations
ACTION_ANNOTATIONS_DIR = "/media/beast-gamma/Media/Datasets/SCAND/ActionAnnotations"
# Where to write the CSV report
OUTPUT_CSV = Path(__file__).parent / "action_annotation_gaps.csv"
# Maximum expected step between annotated frame_idx values
MAX_ALLOWED_STEP = 6


def find_gaps(file_path: str, max_step: int = MAX_ALLOWED_STEP) -> List[Tuple[int, int, int]]:
    """Return gaps where consecutive frame_idx differ by more than max_step."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ann = data.get("annotations_by_stamp", {})
    frame_idxs = sorted(
        int(v["frame_idx"]) for v in ann.values() if isinstance(v, dict) and "frame_idx" in v
    )

    gaps = []
    for i in range(1, len(frame_idxs)):
        prev_idx = frame_idxs[i - 1]
        curr_idx = frame_idxs[i]
        step = curr_idx - prev_idx
        if step > max_step:
            gaps.append((prev_idx, curr_idx, step))
    return gaps


def main():
    json_files = sorted(glob.glob(os.path.join(ACTION_ANNOTATIONS_DIR, "*.json")))
    rows = []

    for jf in json_files:
        bag_name = Path(jf).stem
        gaps = find_gaps(jf)

        gap_list_str = " ".join(f"({a},{b})" for a, b, _ in gaps) if gaps else ""
        total_skip = sum(gap - 1 for _, _, gap in gaps)  # frames skipped between annotated ones

        rows.append(
            {
                "bag": bag_name,
                "gaps": gap_list_str,
                "total_skip": total_skip,
            }
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["bag", "gaps", "total_skip"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Processed {len(json_files)} files.")
    total_gaps = sum(1 for r in rows if r["gaps"])
    print(f"[INFO] Found {total_gaps} files with gaps larger than {MAX_ALLOWED_STEP}.")
    print(f"[INFO] Report written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
