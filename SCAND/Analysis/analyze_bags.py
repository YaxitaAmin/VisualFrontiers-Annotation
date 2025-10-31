import json
import os
from utils import get_existing_bags, get_topics_and_info_from_bag, write_dict_to_json, get_topics_from_bag

ROSBAG_PATH = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/"
ALL_BAGS = "/home/beast-gamma/Documents/GAMMA/Projects/SCAND_Analyse/files.txt"

def extract_topics_from_bags():

    existing_bags = get_existing_bags(ROSBAG_PATH)
    topics_per_bag = {}
    for bag in existing_bags:
        bag_path = os.path.join(ROSBAG_PATH, bag)
        topic_dict = get_topics_and_info_from_bag(bag_path)
        topics_per_bag[bag] = topic_dict
        write_dict_to_json(topics_per_bag, "./topics_per_bag.json")
    # print(len(existing_bags))

def check_annotatability():
    existing_bags = get_existing_bags(ROSBAG_PATH)
    topic_json_path = "./topics_for_project.json"
    skip_json_path = "./bags_to_skip.json"

    with open(topic_json_path, 'r') as f:
        topics_for_project = json.load(f)
    with open(skip_json_path, 'r') as f:
        bags_to_skip = json.load(f)
    
    for bag in existing_bags:
        if "Spot" in bag:
            topics = topics_for_project.get("spot")

        elif "Jackal" in bag:
            topics = topics_for_project.get("jackal")
            
        else:
            print(f"[WARN] Unknown robot in bag name: {bag}")
            continue

        bag_topics = get_topics_from_bag(os.path.join(ROSBAG_PATH, bag))
        # print(f"[INFO] Checking {bag}")
        # print(f"Topics in {bag}: {bag_topics}")

        missing_topics = []
        all_present = True
        for topic_key in topics:
            if topics[topic_key] not in bag_topics:
                missing_topics.append(topics[topic_key])
                all_present = False
        if not all_present:
            print(f"[MISSING] {bag} is missing topics: {missing_topics}")
            skip = bags_to_skip.get(f"{bag}", False)
            if not skip:
                print(f"[ERROR] {bag} is missing required topics and is not marked to skip!")
        else:
            skip = bags_to_skip.get(f"{bag}", False)
            if skip:
                print(f"[WARN] {bag} has all required topics but is marked to skip.")

if __name__ == "__main__":
    check_annotatability()