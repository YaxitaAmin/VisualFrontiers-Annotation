#!/usr/bin/env python3

import csv
import glob
import json
from pathlib import Path
from typing import Dict, Tuple

# Directory containing preference annotation JSON files
PREFERENCE_ANNOTATIONS_DIR = "/media/beast-gamma/Media/Datasets/SCAND/Preference_Annotations"
# Where to write the CSV report
OUTPUT_CSV = Path(__file__).parent / "preference_ranking_report.csv"


def count_preferences(file_path: str) -> Tuple[int, int, Dict[int, int]]:
    """
    Returns (total_preferences, top_not_three, top_choice_counts).

    top_choice_counts tallies how often 0/1/2/3 appear as the first entry in the
    preference list. Only entries with a non-empty list are counted.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations_by_stamp", {})
    top_choice_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_preferences = 0

    for ann in annotations.values():
        pref = ann.get("preference")
        if not isinstance(pref, list) or not pref:
            continue

        top_choice = pref[0]
        if top_choice in top_choice_counts:
            top_choice_counts[top_choice] += 1
        total_preferences += 1

    top_not_three = total_preferences - top_choice_counts[3]
    return total_preferences, top_not_three, top_choice_counts


def main():
    json_files = sorted(glob.glob(f"{PREFERENCE_ANNOTATIONS_DIR}/*.json"))
    rows = []

    for jf in json_files:
        bag_name = Path(jf).stem
        total, not_three, counts = count_preferences(jf)
        rows.append(
            {
                "bag": bag_name,
                "total_preferences": total,
                "top_not_three": not_three,
                "top_0": counts[0],
                "top_1": counts[1],
                "top_2": counts[2],
                "top_3": counts[3],
            }
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "bag",
                "total_preferences",
                "top_not_three",
                "top_0",
                "top_1",
                "top_2",
                "top_3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Processed {len(json_files)} files.")
    print(f"[INFO] Report written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
