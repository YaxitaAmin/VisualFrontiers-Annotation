from utils import get_existing_bags, get_topics_and_info_from_bag, write_dict_to_json, get_topics_from_bag
import rosbag
import json
import os
import csv
import numpy as np

ROSBAG_PATH = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/"
OUT_CSV = "./lateral_v_results.csv"

def check_lateral_v():
    existing_bags = get_existing_bags(ROSBAG_PATH)
    topic_json_path = "./topics_for_project.json"

    with open(topic_json_path, 'r') as f:
        topics_for_project = json.load(f)
    
    results = []
    for bag in existing_bags:
        if "Jackal" in bag:
            odom_topic = topics_for_project.get("jackal").get("odom")
        elif "Spot" in bag:
            odom_topic = topics_for_project.get("spot").get("odom")
        else:
            print(f"[WARN] Unknown robot in bag name: {bag}")
            continue
    
        vy_values = []
        try:
            with rosbag.Bag(os.path.join(ROSBAG_PATH, bag), 'r') as b:
                for _, msg, _ in b.read_messages(topics=[odom_topic]):
                    try:
                        vy = msg.twist.twist.linear.y
                        if np.abs(vy) > 0:
                            vy_values.append(vy)

                    except AttributeError:
                        continue
        except Exception as e:
            print(f"[ERROR] Failed to read {bag}: {e}")
            continue
        
        if len(vy_values) > 0:
            count_vy = len(vy_values)
            max_vy = float(np.max(vy_values))
            mean_vy = float(np.mean(vy_values))
            print(f"[INFO] {bag}: Detected vy>0 ({count_vy} frames, max={max_vy:.3f}, mean={mean_vy:.3f})")

            results.append({
            "bag_name": bag,
            "count_vy_positive": count_vy,
            "max_vy": round(max_vy, 6),
            "mean_vy": round(mean_vy, 6)
            })
        else:
            count_vy = 0
            max_vy = 0.0
            mean_vy = 0.0
            print(f"[INFO] {bag}: No lateral vy>0 found.")

    # Write CSV summary
    with open(OUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ["bag_name", "count_vy_positive", "max_vy", "mean_vy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[OK] Results written to {OUT_CSV}")
    return results

if __name__ == "__main__":
    check_lateral_v()