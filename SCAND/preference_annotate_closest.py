#!/usr/bin/env python3

import os
import csv
import glob
import math
import random
from pathlib import Path
from typing import Optional, Tuple
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass

# --- ROS1 imports ---
import rosbag
from cv_bridge import CvBridge

from preference_annotate import TemporalAnnotator as BaseTemporalAnnotator
from preference_annotate import FrameItem, PathItem

# Colors (BGR)
COLOR_PATH_ONE = (0, 0, 255)    # RED
COLOR_PATH_TWO = (0, 255, 0)    # GREEN
CLOSER_COLOR = (0, 255, 255)    # YELLOW

BAD_BOTH = 500
TIE_BOTH = 404

# ===========================
# Main Annotator
# ===========================
class TemporalAnnotatorClosest(BaseTemporalAnnotator):
    def __init__(self, bag_path, calib_path, topics_path, annotations_root, expert_action_annotation_dir, lookahead=5, num_keypoints=5, max_deviation=1.5):
        super().__init__(bag_path, calib_path, topics_path, annotations_root, expert_action_annotation_dir, lookahead, num_keypoints, max_deviation)
        self.timed_exit = True

    def check_closer_to_goal_direction(self, pitem_1: PathItem, pitem_2: PathItem) -> Tuple[int, int]:
        # Determine which path is closer to the annotator goal
        if self.annotator_goal is None:
            True
        p1_last = pitem_1.path_points[-1][1]
        p2_last = pitem_2.path_points[-1][1]

        dist_1 = np.linalg.norm(p1_last - self.annotator_goal[1])
        dist_2 = np.linalg.norm(p2_last - self.annotator_goal[1])

        return dist_1<dist_2

    def _compute_ranking(self):
        """
        Preference ordering:
          3: annotator-selected path (always best)
          0: expert/executed path (kept second as a negative example)
          1/2: whichever offset path ends closer to the annotator goal, then the other
        """
        n_paths = len(self.paths)
        if self.annotator_goal is None or n_paths < 4:
            return [3, 0, 1, 2]

        rankings = [3, 0]
        closer_is_one = self.check_closer_to_goal_direction(self.paths[1], self.paths[2])

        if closer_is_one:
            rankings.extend([1, 2])
        else:
            rankings.extend([2, 1])
        return rankings

    def process_bag(self, undersampling_factor):
        
        # print(self.timestamps[:10])
        count = 0
        skip_count = 0
        timestamp_counter = 0
        with rosbag.Bag(self.bag_path, "r") as bag:
            pos_defined = False
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic, self.odom_topic])):
                # print(int(str(t)), int(self.timestamps[timestamp_counter]), timestamp_counter, len(self.timestamps))
                if(int(str(t)) > int(self.timestamps[timestamp_counter]) and not pos_defined):
                    timestamp_counter += 1
                    skip_count += 1
                if topic == self.odom_topic:
                    pos, v, w, rot, yaw = self.process_odom(msg)
                    pos_defined = True
                elif topic == self.image_topic:
                    cv_img = self.process_image(msg)

                    if pos_defined and str(t) == str(self.timestamps[timestamp_counter]):
                        self.frames.append(FrameItem(idx=count, stamp=t, img=cv_img, position=pos, velocity=v, omega=w, rotation=rot, yaw = yaw))
                        timestamp_counter += 1
                        count+=1       
                if timestamp_counter >= len(self.timestamps):
                    break 
        print(f"[INFO] Loaded {len(self.frames)} frames from bag after skipping {skip_count} frames. There were {len(self.timestamps)} timestamps in total.")
        if not self.frames:
            print("[WARN] No frames after undersampling.")
            return

        try:
            for i in range(len(self.frames)):
                fr = self.frames[i]
                self.frame_idx = i
                self.frame_stamp = fr.stamp
                self.current_img = fr.img
                self.current_pos = list(fr.position)
                self.current_yaw = fr.yaw

                self.reset()

                self.path, self.yaws, self.cum_dists = self.compute_path()
                self.paths.append(self.create_path_item(self.path, self.yaws))

                # fast-skip degenerate/off-view frames
                if (self.path is None or self.path.shape[0] < 2 or
                    self.cum_dists is None or self.cum_dists.size == 0 or
                    self.cum_dists[-1] <= 1.0):
                    continue

                left_offset_path, right_offset_path, annotator_path = self.compute_comparison_paths()
                self.comparison_paths = [left_offset_path, right_offset_path, annotator_path] #[1,2,3,4]

                for p in self.comparison_paths:
                    self.paths.append(self.create_path_item(p, yaws=None))
                
                self.log_frame()
        finally:
            self._close_bag_doc()

if __name__ == "__main__":

    # ===========================
    # Configs
    # ===========================

    bag_dir = "/media/beast-gamma/Media/Datasets/SCAND/rosbags"   # Point to path with rosbags being annotated for the day
    expert_action_annotation_dir = "/media/beast-gamma/Media/Datasets/SCAND/ActionAnnotations"
    annotations_root = "./Annotations_closest"
    calib_path = "./tf.json"
    skip_json_path = "./bags_to_skip.json"
    topic_json_path = "./topics_for_project.json"

    fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
    undersampling_factor = 6
    lookahead = 7 #in m if the lookahead is distance , in s if the lookahead is time. 
    num_keypoints = 5
    max_deviation = 1.5

    bag_files = sorted(glob.glob(os.path.join(bag_dir, "*.bag")))

    with open(skip_json_path, 'r') as f:
        bags_to_skip = json.load(f)

    if not bag_files:
        print(f"[ERROR] No .bag files found in {bag_dir}")

    for bp in bag_files:
        if bags_to_skip.get(os.path.basename(bp), False):
            print(f"[INFO] Skipping {bp}")
            continue
        
        print(f"[INFO] Processing {bp}")
        annotator = TemporalAnnotatorClosest(bp, calib_path, topic_json_path, annotations_root, expert_action_annotation_dir, lookahead, num_keypoints, max_deviation)
        annotator.process_bag(undersampling_factor)

    print(f"\n[DONE] Annotations written to {annotator.output_path}")
