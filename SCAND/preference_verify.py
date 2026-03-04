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

from vis_utils import camray_to_ground_in_base, transform_points, load_calibration, \
    make_corridor_polygon, draw_polyline, draw_corridor, project_points_cam, add_first_point, \
    project_clip, make_corridor_polygon_from_cam_lines, clean_2d
from traj_utils import solve_arc_from_point, arc_to_traj, make_offset_paths, create_yaws_from_path, \
    make_offset_path_to_point
from utils import get_topics_from_bag

# Colors (BGR)
COLOR_PATH_ONE = (0, 0, 255)    # RED
COLOR_PATH_TWO = (0, 255, 0)    # GREEN
CLOSER_COLOR = (0, 255, 255)    # YELLOW

BAD_BOTH = 500
TIE_BOTH = 404

# ===========================
# Camera & Geometry helpers
# ===========================

@dataclass
class PathItem:
    path_points: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    polygon: np.ndarray

# ===========================
# Main Annotator
# ===========================
class TemporalAnnotationVerifier:
    def __init__(self, bag_path, calib_path, topics_path, annotations_root):
        self.bag_path = bag_path
        self.bag_name = Path(bag_path).name
        self.needs_correction = False
        stem = Path(self.bag_name).stem
        self.output_path = os.path.join(annotations_root, f"{stem}_verified.json")
        self.preference_annotation_path = os.path.join(annotations_root, f"{stem}.json")

        with open(self.preference_annotation_path, 'r') as f:
            self.preference_annotations = json.load(f)

        with open(topics_path, 'r') as f:
            topics = json.load(f)
        
        if "Jackal" in self.bag_name:
            self.K, self.dist, self.T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="jackal")
            self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
            mode = "jackal"
        elif "Spot" in self.bag_name:
            self.K, self.dist, self.T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="spot")
            self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
            mode = "spot"
            self.needs_correction = True
        else:
            raise Exception
        
        self.image_topic = topics.get(mode).get("camera")
        self.robot_width = topics.get(mode).get("width")

        print(f"[INFO] Using image topic: {self.image_topic}, robot width: {self.robot_width}")
 
        self.bridge = CvBridge()

        self.window = "SCAND Temporal Verifier"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        self.current_img = None
        self.current_img_show = None        

        self.frame_idx = -1
        self.frame_stamp = None

        self.bag_doc = self._open_bag_doc()
        self.timestamps = self._get_timestamps_from_annotations()


        self.paths : list[PathItem] = []

        self.active_pairs = []
        self.choices = []
        self.active_pair_idx = 0
        self.prefs_this_frame = []
        self._finalize_and_advance = False

        self.first_frame = True
        self.annotator_goal = None
        self.initial_choice = None

    def _get_timestamps_from_annotations(self):
        timestamps = []
        for key in self.preference_annotations.get("annotations_by_stamp", {}).keys():
            timestamps.append(int(key))
        return timestamps
    def _open_bag_doc(self):
        bag_doc = {
            "bag": self.bag_name,
            "image_topic": self.image_topic,
            "annotations_by_stamp": {}
        }

        return bag_doc

    def _save_pairwise_and_ranking(self, stamp_str: str):
        root = self.preference_annotations
        if "annotations_by_stamp" not in root:
            root["annotations_by_stamp"] = {}
        if stamp_str not in root["annotations_by_stamp"]:
            root["annotations_by_stamp"][stamp_str] = {}

        entry = root["annotations_by_stamp"][stamp_str]
        entry["pairwise"] = self.prefs_this_frame
        entry["preference"] = self._compute_ranking()

        with open(self.preference_annotation_path, "w", encoding="utf-8") as f:
            json.dump(root, f, ensure_ascii=False, indent=2)

    def _close_bag_doc(self):
        if self.bag_doc is not None:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.bag_doc, f, ensure_ascii=False, indent=2)
            self.bag_doc = None

    def _load_paths_for_stamp(self, stamp_str: str):
        ann = self.preference_annotations.get("annotations_by_stamp", {}).get(stamp_str)
        if not ann:
            return False
        saved_paths = ann.get("paths", {})
        if not saved_paths:
            print(f"[WARN] No 'paths' saved for {stamp_str}; skipping.")
            return False

        # Rebuild paths in numeric order "0","1","2",...
        idxs = sorted(int(k) for k in saved_paths.keys())
        self.paths = []
        self.active_pairs = []
        self.choices = []

        for k in idxs:
            p = saved_paths[str(k)]
            self.paths.append(PathItem(
                path_points=np.asarray(p["points"], dtype=float),
                left_boundary=np.asarray(p["left_boundary"], dtype=float),
                right_boundary=np.asarray(p["right_boundary"], dtype=float),
                polygon=None
            ))

        self.active_pair_idx = 0

        # Seed existing pairwise (so we can pre-highlight)
        self.prefs_this_frame = []
        for e in ann.get("pairwise", []):
            i, j = int(e["pair"][0]), int(e["pair"][1])
            self.choices.append(int(e["choice"]))
            self.active_pairs.append((i,j))
            self.prefs_this_frame.append({
                "pair": [i, j],
                "choice": int(e["choice"]),
                "meaning": e.get("meaning", "picked_index")
            })
        return True
    
    def draw(self):
        if self.current_img is None or not self.paths:
            return
        img = self.current_img.copy()
        img_h, img_w = img.shape[:2]
        # print(img_h, img_w)
        # pick current pair; fallback to (0,1)
        i, j = self.active_pairs[self.active_pair_idx]
        chosen = self.choices[self.active_pair_idx]

        def draw_one(pitem, color, chosen):
            points_2d = clean_2d(project_clip(pitem.path_points, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            left_2d   = clean_2d(project_clip(pitem.left_boundary,  self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            right_2d  = clean_2d(project_clip(pitem.right_boundary, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            # points_2d = project_clip(pitem.path_points, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            # left_2d   = project_clip(pitem.left_boundary,  self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            # right_2d  = project_clip(pitem.right_boundary, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            poly_2d   = make_corridor_polygon_from_cam_lines(left_2d, right_2d)
            draw_polyline(img, points_2d, 2, color)

            if chosen:
                draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.35, fill_color=color, edge_color=CLOSER_COLOR, edge_thickness=2)
            else:
                draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.15, fill_color=color, edge_color=color, edge_thickness=2)

        self.active_pairs[self.active_pair_idx] = (i, j)
        
        if i < len(self.paths): draw_one(self.paths[i], COLOR_PATH_ONE, i == chosen)     # RED
        if j < len(self.paths): draw_one(self.paths[j], COLOR_PATH_TWO, j == chosen)     # GREEN

        label = f"Compare ({i},{j})  [1]=RED  [2]=GREEN [3]=Both bad  [4]=No pref ({self.active_pair_idx+1}/{len(self.active_pairs)} Frame: {self.frame_idx}/{len(self.timestamps)-1})"
        cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

        self.current_img_show = img
        cv2.imshow(self.window, self.current_img_show)

        self.initial_choice = i if i == chosen else (j if j == chosen else BAD_BOTH if chosen == BAD_BOTH else TIE_BOTH if chosen == TIE_BOTH else None)
    
    def _path_item_to_dict(self, pitem: PathItem, stamp_obj):
        # stamp as a string for readability; also give a per-point parallel stamp list if you want
        stamp_str = str(stamp_obj)
        return {
            "points": pitem.path_points.tolist(),
            "left_boundary": pitem.left_boundary.tolist(),
            "right_boundary": pitem.right_boundary.tolist(),
            "timestamp": stamp_str
        }
    
    def reset(self):
        self.paths = []
        self.choices = []
        self.prefs_this_frame = []
        self._finalize_and_advance = False
        self.active_pairs = []
        self.active_pair_idx = 0

    def log_frame(self):
        if self.bag_doc is None or self.frame_stamp is None:
            return
        stamp_key = str(self.frame_stamp)

        paths_dict = {}
        for idx, pitem in enumerate(self.paths[:4]):  # ensure 0..4 exist
            paths_dict[str(idx)] = self._path_item_to_dict(pitem, self.frame_stamp)

        ranking = self._compute_ranking()

        self.bag_doc["annotations_by_stamp"][stamp_key] = {
            "frame_idx": int(self.frame_idx),
            "robot_width": self.robot_width,
            "paths": paths_dict,
            "preference": ranking,
            "pairwise": self.prefs_this_frame
        }

    def process_image(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        return cv_img

    def _compute_ranking(self):
        n = min(len(self.paths), 4)     # you build 4
        idxs = list(range(n))
        wins = {k: 0 for k in idxs}
        for e in self.prefs_this_frame:
            i, j = e["pair"]; c = e["choice"]
            if i in wins and j in wins and c in wins:
                wins[c] += 1
            if c == BAD_BOTH:
                # both bad
                wins[i] -= 1
                wins[j] -= 1
            if c == TIE_BOTH:
                # no pref
                wins[i] += 0.5
                wins[j] += 0.5

        order = sorted(idxs, key=lambda k: (-wins[k], k))  # tie -> lower index
        return order
    
    def _record_choice(self, chosen_path: int):
        if self.active_pair_idx < 0 or self.active_pair_idx >= len(self.active_pairs):
            return
        pair = self.active_pairs[self.active_pair_idx]
        self.prefs_this_frame[self.active_pair_idx]["choice"] = int(chosen_path)
        self.prefs_this_frame[self.active_pair_idx]["meaning"] = "both_bad" if chosen_path == BAD_BOTH else ("no_preference" if chosen_path == TIE_BOTH else "picked_index")

        self.active_pair_idx += 1
        if self.active_pair_idx == len(self.active_pairs):
            self._finalize_and_advance = True

    def process_bag(self, undersampling_factor):
        
        # print(self.timestamps[:10])
        count = 0
        try:
            with rosbag.Bag(self.bag_path, "r") as bag:
                for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic])):
                    if count%undersampling_factor==0 and self.preference_annotations.get("annotations_by_stamp", {}).get(str(t), None) is not None:
                        if topic == self.image_topic:
                            self.frame_idx = count
                            self.frame_stamp = str(t)
                            self.current_img = self.process_image(msg)
                            count+=1
                        else:
                            print("[WARN] timestamp in annotations but no image found.")

                    else:
                        count+=1
                        continue
                    
                    paths_exist = self._load_paths_for_stamp(str(t))
                    
                    if not paths_exist:
                        print(f"[WARN] No paths for timestamp {t}; skipping.")
                        continue

                    while True:
                        self.draw()
                        key = cv2.waitKey(0) & 0xFF
                            
                        if key in (ord('q'), 27):
                            print("[INFO] Quit requested.")
                            cv2.destroyAllWindows()
                            return

                        cur = self.active_pairs[self.active_pair_idx] if self.active_pair_idx < len(self.active_pairs) else None
                        if cur is None:
                            continue

                        if key == ord('1'):
                            self._record_choice(cur[0])       # RED
                        elif key == ord('2'):
                            self._record_choice(cur[1])       # GREEN
                        elif key == ord('3'):
                            self._record_choice(BAD_BOTH)
                        elif key == ord('4'):
                            self._record_choice(TIE_BOTH)
                        elif key == 83:  # Right Arrow
                            if self.initial_choice not in cur and self.initial_choice in (BAD_BOTH, TIE_BOTH):
                                self._record_choice(self.initial_choice)
                            else:
                                self._record_choice(self.initial_choice)  # map 0/1 → global index
                        elif key == 81:  # Left Arrow (optional: implement per-stamp backtrack)
                            print("[INFO] Back one (not implemented across stamps).")
                            break
                        else:
                            continue

                        if self._finalize_and_advance:
                            # Write back only pairwise + preference to the SAME json
                            self.log_frame()
                            self.reset()
                            break
        finally:
            self._close_bag_doc()

if __name__ == "__main__":

    # ===========================
    # Configs
    # ===========================

    bag_dir = "/media/beast-gamma/Media/Datasets/SCAND/annt"   # Point to path with rosbags being annotated for the day
    annotations_root = "./Annotations_closest"
    calib_path = "./tf.json"
    skip_json_path = "./bags_to_skip.json"
    topic_json_path = "./topics_for_project.json"

    fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
    undersampling_factor = 6

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
        annotator = TemporalAnnotationVerifier(bp, calib_path, topic_json_path, annotations_root)
        annotator.process_bag(undersampling_factor)

    print(f"\n[DONE] Annotations written to {annotator.output_path}")
