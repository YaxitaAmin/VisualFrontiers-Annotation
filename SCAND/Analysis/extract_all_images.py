import os
import rosbag
from cv_bridge import CvBridge
import cv2
import json

bag_file_root = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/"
output_root = "/media/beast-gamma/Media/Datasets/SCAND/images"
skip_json_path = "./bags_to_skip.json"
topics_for_project_path = "./topics_for_project.json"

def extract_all_images():

    with open(skip_json_path, 'r') as f:
        bags_to_skip = json.load(f)
    
    with open(topics_for_project_path, 'r') as f:
        topics_for_project = json.load(f)

    bridge = CvBridge()
    bag_files = sorted([f for f in os.listdir(bag_file_root) if f.endswith('.bag')])
    for bag_file in bag_files:
        if bags_to_skip.get(bag_file, False):
            print(f"[INFO] Skipping {bag_file}")
            continue

        bag_path = os.path.join(bag_file_root, bag_file)
        output_dir = os.path.join(output_root, os.path.splitext(bag_file)[0])
        os.makedirs(output_dir, exist_ok=True)

        print(f"[INFO] Processing {bag_file}")
        with rosbag.Bag(bag_path, 'r') as bag:
            img_count = 0
            if "Jackal" in bag_file:
                image_topic = "/camera/rgb/image_raw/compressed"
            elif "Spot" in bag_file:
                image_topic = "/image_raw/compressed"
            
            for topic, msg, t in bag.read_messages(topics=[image_topic]):
                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                img_filename = os.path.join(output_dir, f"img_{str(t)}.png")
                cv2.imwrite(img_filename, cv_img)
                img_count += 1
            print(f"[INFO] Extracted {img_count} images from {bag_file}")

if __name__ == "__main__":
    extract_all_images()