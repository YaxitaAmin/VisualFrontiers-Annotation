import os
import sys

ROSBAG_ROOT = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/"
ALL_BAGS = "/home/beast-gamma/Documents/GAMMA/Projects/SCAND_Analyse/files.txt"

def get_existing_bags(path):
    existing_bags = []

    for root, dirs, files in os.walk(path):
        print(len(files))
        for filename in files:
            if filename.endswith(".bag"):
                existing_bags.append(filename)

    return existing_bags

def read_all_bags(file_path):
    with open(file_path, 'r') as file:
        all_bags = [line.strip() for line in file.readlines()]
    return all_bags

def is_bag_corrupted(bag_path):
    """
    Try to open the bag file with rosbag.
    Returns True if corrupted, False if OK.
    """
    bag_path = os.path.join(ROSBAG_ROOT, bag_path)
    try:
        import rosbag  # ROS1
        with rosbag.Bag(bag_path, "r") as bag:
            bag.get_type_and_topic_info()
        return False  # OK
    except Exception as e:
        print(f"[CORRUPTED] {bag_path}: {e}")
        return True
    
def main():

    existing_bags = get_existing_bags(ROSBAG_ROOT)
    all_bags = read_all_bags(ALL_BAGS)
    missing_bags = []
    # print(existing_bags, len(existing_bags))
    # print(all_bags, len(all_bags))
    count = 0
    for bag in all_bags:
        print(f"Checking {bag}...")
        if bag not in existing_bags:
            count += 1
            print(f"{bag}, count: {count}")
            missing_bags.append(bag)
        elif is_bag_corrupted(bag): 
            count += 1
            print(f"{bag}, count: {count}")
            missing_bags.append(bag)
        else:
            pass
    
    print(f"Total missing or corrupted bags: {len(missing_bags)}")

    for bag in missing_bags:
        with open("missing_bags.txt", "a") as f:
            f.write(f"{bag}\n")

if __name__ == "__main__":
    main()