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
class FrameItem:
    idx: int
    stamp: object   # rospy.Time
    img: np.ndarray
    position: np.ndarray 
    velocity: float
    omega: float
    rotation: np.ndarray
    yaw: float

@dataclass
class PathItem:
    path_points: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    polygon: np.ndarray

# ===========================
# Main Annotator
# ===========================
class TemporalAnnotator:
    def __init__(self, bag_path, calib_path, topics_path, annotations_root, expert_action_annotation_dir, lookahead=5, num_keypoints=5, max_deviation=1.5):
        self.bag_path = bag_path
        self.bag_name = Path(bag_path).name
        self.needs_correction = False
        stem = Path(self.bag_name).stem
        self.output_path = os.path.join(annotations_root, f"{stem}.json")
        fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0

        self.expert_action_annotation_dir = os.path.join(expert_action_annotation_dir, self.bag_name.replace(".bag", ".json"))
        print(f"[INFO] Expert action annotations path: {self.expert_action_annotation_dir}")
        with open(topics_path, 'r') as f:
            topics = json.load(f)
        
        try:
            with open(self.expert_action_annotation_dir, 'r') as f:
                self.action_annotations = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load expert action annotations from {self.expert_action_annotation_dir}: {e}")
            raise e
        
        self.timestamps = self._get_timestamps_from_expert_annotations()
        print(len(self.timestamps), "timestamps from expert annotations loaded.")
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
        self.odom_topic = topics.get(mode).get("odom")
        self.robot_width = topics.get(mode).get("width")
        self.lookahead = lookahead
        self.lookaheads = [lookahead, lookahead, lookahead, lookahead, lookahead, lookahead, lookahead, lookahead, lookahead, lookahead]
        self.keypoint_res = lookahead/num_keypoints
        self.v_mean = [1, 1, 1, 1, 1]
        self.max_duration = 2.4
        self.L_max = 8.0

        print(f"[INFO] Using image topic: {self.image_topic}, control topic: {self.odom_topic}, robot width: {self.robot_width}")
 
        self.bridge = CvBridge()

        self.window = "SCAND Temporal Annotator"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        self.current_img = None
        self.current_img_show = None        

        self.frame_idx = -1
        self.frame_stamp = None

        self.bag_doc = self._open_bag_doc()
        self.frames : list[FrameItem] = []

        self.path = None
        self.yaws = None
        self.cum_dists = None

        self.comparison_paths = []
        self.max_deviation = max_deviation

        self.paths : list[PathItem] = []
        self.comp_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

        self.active_pairs = list(self.comp_pairs)
        self.active_pair_idx = 0
        self.prefs_this_frame = []
        self._finalize_and_advance = False

        self.first_frame = True
        self.annotator_goal = None
        self.initial_choice = None
        self.timed_exit = False
        self.stop = False

        self.current_pos = None
        self.current_yaw = None

    def _get_timestamps_from_expert_annotations(self):
        timestamps = []
        for key in self.action_annotations.get("annotations_by_stamp", {}).keys():
            timestamps.append(int(key))
        return timestamps

    def _open_bag_doc(self):
        bag_doc = {
            "bag": self.bag_name,
            "image_topic": self.image_topic,
            "annotations_by_stamp": {}
        }

        return bag_doc

    def _close_bag_doc(self):
        if self.bag_doc is not None:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.bag_doc, f, ensure_ascii=False, indent=2)
            self.bag_doc = None

    def draw(self):
        if self.current_img is None or not self.paths:
            return
        img = self.current_img.copy()
        img_h, img_w = img.shape[:2]
        # print(img_h, img_w)
        # pick current pair; fallback to (0,1)
        if 0 <= self.active_pair_idx < len(self.active_pairs):
            i, j = self.active_pairs[self.active_pair_idx]
        else:
            i, j = (0, 1)

        def check_left_right(pitem_1: PathItem, pitem_2: PathItem) -> Tuple[int, int]:
            # Determine which path is left/right based on the first point
            p1_last = pitem_1.path_points[-1]
            p2_last = pitem_2.path_points[-1]

            if p1_last[1] > p2_last[1]:
                return (i, j)
            else:
                return (j, i)
        
        def check_closer_to_goal_direction(pitem_1: PathItem, pitem_2: PathItem) -> Tuple[int, int]:
            # Determine which path is closer to the annotator goal
            if self.annotator_goal is None:
                True
            p1_last = pitem_1.path_points[-1][1]
            p2_last = pitem_2.path_points[-1][1]

            dist_1 = np.linalg.norm(p1_last - self.annotator_goal[1])
            dist_2 = np.linalg.norm(p2_last - self.annotator_goal[1])

            return dist_1<dist_2
            
        def draw_one(pitem, color, closer):
            points_2d = clean_2d(project_clip(pitem.path_points, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            left_2d   = clean_2d(project_clip(pitem.left_boundary,  self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            right_2d  = clean_2d(project_clip(pitem.right_boundary, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            # points_2d = project_clip(pitem.path_points, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            # left_2d   = project_clip(pitem.left_boundary,  self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            # right_2d  = project_clip(pitem.right_boundary, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            poly_2d   = make_corridor_polygon_from_cam_lines(left_2d, right_2d)
            draw_polyline(img, points_2d, 2, color)

            if closer:
                draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.35, fill_color=color, edge_color=CLOSER_COLOR, edge_thickness=2)
            else:
                draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.15, fill_color=color, edge_color=color, edge_thickness=2)

        i, j = check_left_right(self.paths[i], self.paths[j])
        closer_1 = check_closer_to_goal_direction(self.paths[i], self.paths[j])
        closer_2 = not closer_1

        self.active_pairs[self.active_pair_idx] = (i, j)
        
        if i < len(self.paths): draw_one(self.paths[i], COLOR_PATH_ONE, closer_1)     # RED
        if j < len(self.paths): draw_one(self.paths[j], COLOR_PATH_TWO, closer_2)     # GREEN

        label = f"Compare ({i},{j})  [1]=RED  [2]=GREEN [3]=Both bad  [4]=No pref ({self.active_pair_idx+1}/{len(self.active_pairs)} Frame: {self.frame_idx}/{len(self.frames)})"
        cv2.putText(img, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

        self.current_img_show = img
        cv2.imshow(self.window, self.current_img_show)

        if closer_1:
            self.initial_choice = 0
        elif closer_2:
            self.initial_choice = 1
        
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
        self.prefs_this_frame = []
        self._finalize_and_advance = False
        self.active_pairs = list(self.comp_pairs)  # phase 1 again
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
            "pairwise": self.prefs_this_frame, 
            "position": self.current_pos,
            "yaw" : self.current_yaw,
            "stop": self.stop
        }

    def process_odom(self, msg):
        
        quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        if self.needs_correction:
            current_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            velocity_robot_frame = np.linalg.inv(rotation_matrix) @ current_vel

            v = velocity_robot_frame[0]
        else:
            v = msg.twist.twist.linear.x
        
        yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        w = msg.twist.twist.angular.z

        return pos,v,w, rotation_matrix, yaw

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
    
    def _record_choice(self, chosen_idx: int):
        if self.active_pair_idx < 0 or self.active_pair_idx >= len(self.active_pairs):
            return
        pair = self.active_pairs[self.active_pair_idx]
        
        self.prefs_this_frame.append({
            "pair": [int(pair[0]), int(pair[1])], 
            "choice": int(chosen_idx),
            "meaning": "both_bad" if chosen_idx == BAD_BOTH else ("no_preference" if chosen_idx == TIE_BOTH else "picked_index")
            })

        self.active_pair_idx += 1

        # If we just finished the current list of pairs, decide next phase
        if self.active_pair_idx == len(self.active_pairs):
            self._finalize_and_advance = True

    def create_path_item(self, path_points: np.ndarray, yaws: np.ndarray) -> PathItem:

        if yaws is None:
            yaws = create_yaws_from_path(path_points)

        left_b, right_b, poly_b = make_corridor_polygon(path_points, yaws, self.robot_width)
        
        return PathItem(path_points=path_points, left_boundary=left_b, right_boundary=right_b, polygon=poly_b)

    def compute_comparison_paths(self):
        max_offset = max(self.max_deviation * self.cum_dists[-1]/self.L_max, self.robot_width)
        offset = np.random.uniform(self.robot_width/2, max_offset)
        if self.cum_dists[-1] == 0 or self.path is None or self.path.shape[0] < 2:
            return np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([[0, 0, 0]])

        offset_ratios = self.cum_dists / self.cum_dists[-1]
        offsets = offset * (offset_ratios)

        left_offset_path, right_offset_path = make_offset_paths(self.path, self.yaws, offsets)

        #Hann conv offsets
        # offset = np.random.uniform(self.robot_width/2, self.robot_width)
        # base = 0.5 * (1.0 - np.cos(2.0 * np.pi * offset_ratios))
        # gamma = 1.0
        # w = base ** gamma
        # conv_offsets = offset * w

        # # print(conv_offsets)
        # left_conv_path, right_conv_path = make_offset_paths(self.path, self.yaws, conv_offsets)

        stop_flag = self.action_annotations.get("annotations_by_stamp", {}).get(str(self.frame_stamp), {}).get("stop", False)

        if not stop_flag:
            annotator_goal = self.action_annotations.get("annotations_by_stamp", {}).get(str(self.frame_stamp), {}).get("goal_base", None)

            if annotator_goal is None:
                raise Exception

            self.annotator_goal = np.array([annotator_goal['x'], annotator_goal['y'], annotator_goal['z']])
            expert_path = make_offset_path_to_point(self.path, self.yaws, self.annotator_goal, self.cum_dists)
            self.stop = False
        else:
            # print("HMm")
            expert_path = np.zeros((10,3), dtype=np.float32)
            self.stop = True
        return left_offset_path, right_offset_path, expert_path
    
    def compute_path(self):

        current_pos = self.frames[self.frame_idx].position
        current_yaw = self.frames[self.frame_idx].yaw
        # rot = np.linalg.inv(self.frames[self.frame_idx].rotation)
        v = self.frames[self.frame_idx].velocity
        w = self.frames[self.frame_idx].omega

        rot = self.frames[self.frame_idx].rotation

        frame_pointer = self.frame_idx + 1
        keypoint_count = 0

        arc_length = 0

        if self.first_frame:
            self.lookahead = self.dynamic_lookahead(v, w)
            self.lookaheads = [self.lookahead, self.lookahead, self.lookahead, self.lookahead, self.lookahead, self.lookahead, self.lookahead, self.lookahead, self.lookahead, self.lookahead]
            self.first_frame = False
        else:
            self.lookahead = self.dynamic_lookahead(v, w)
            self.lookaheads.pop(0)
            self.lookaheads.append(self.lookahead)

        self.lookahead = sorted(self.lookaheads)[len(self.lookaheads)//2]

        path = [current_pos]
        yaws = [current_yaw]
        cum_dists = [0]
        t1 = self.frames[self.frame_idx].stamp
        while True:

            # print(arc_length, self.lookahead)11
            if frame_pointer > len(self.frames) - 1 :
                # print("Hmm")
                break

            pos_w = self.frames[frame_pointer].position
            yaw_t = self.frames[frame_pointer].yaw
            diff = (pos_w - path[-1])         #robots frame pos

            distance = np.linalg.norm(diff)
            arc_length += distance
            t2 = self.frames[frame_pointer].stamp
            # print(distance)
            # print((t2 - t1).to_sec())
            if arc_length > self.lookahead or (float((t2 - t1).to_sec()) > self.max_duration and self.timed_exit):
                break
            elif arc_length > (keypoint_count+1)*self.keypoint_res:
                keypoint_count+=1

            cum_dists.append(arc_length)
            path.append(pos_w)
            yaws.append(yaw_t)
            frame_pointer+=1
        
        path = np.array(path)
        yaws = np.array(yaws)
        cum_dists = np.array(cum_dists)

        path = path - current_pos
        yaws = yaws - current_yaw

        path_r = path@rot
        valid = path_r[:,0]>=0
        path_r = path_r[valid]  #only forward points
        path_r[:,2] = 0.0
        yaws = yaws[valid]
        cum_dists = cum_dists[valid]

        return path_r, yaws, cum_dists
    
    def dynamic_lookahead(self, v: float, w: float,
                      T: float = 4.0,           # time headway [s]
                      a_brake: float = 2.5,      # comfortable decel [m/s^2]
                      L_min: float = 2.0,        # never smaller than this [m]
                      L_max: float = 8.0,       # never larger than this [m]
                      kappa_gain: float = 2.0,   # how much to shrink on curves
                      eps: float = 1e-3) -> float:
        """Return meters of lookahead based on speed & curvature."""

        if self.first_frame:
            self.first_frame = False
            self.v_mean = [v, v, v, v, v, v, v, v, v, v]

        v = max(0.0, v)                              # ignore reverse for now
        self.v_mean.pop(0)
        self.v_mean.append(v)
        v = sorted(self.v_mean)[len(self.v_mean)//2]
        L_time  = v * T
        L_stop  = (v * v) / (2.0 * max(a_brake, 1e-6))
        L_base  = max(L_time, L_stop, L_min)

        # curvature penalty (shorter on tight turns)
        if v > eps:
            kappa = abs(w) / v                       # 1/m
            curve_factor = 1.0 / (1.0 + kappa_gain * kappa)
            L_base *= curve_factor

        return max(L_min, min(L_base, L_max))
    
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
        print(f"[INFO] Loaded {len(self.frames)} frames from bag after skipping {skip_count} frames.")
        if not self.frames:
            print("[WARN] No frames after undersampling.")
            return

        i = 0
        try:
            while 0 <= i < len(self.frames):
                fr = self.frames[i]
                self.frame_idx = i
                self.frame_stamp = fr.stamp
                self.current_img = fr.img
                self.reset()

                self.path, self.yaws, self.cum_dists = self.compute_path()
                self.paths.append(self.create_path_item(self.path, self.yaws))

                # fast-skip degenerate/off-view frames
                if (self.path is None or self.path.shape[0] < 2 or
                    self.cum_dists is None or self.cum_dists.size == 0 or
                    self.cum_dists[-1] <= 1.0):
                    i += 1
                    continue

                left_offset_path, right_offset_path, annotator_path = self.compute_comparison_paths()
                self.comparison_paths = [left_offset_path, right_offset_path, annotator_path] #[1,2,3,4]

                for p in self.comparison_paths:
                    self.paths.append(self.create_path_item(p, yaws=None))

                while True:
                    self.draw()
                    key = cv2.waitKey(0) & 0xFF
                        
                    if key in (ord('q'), 27):   # q or ESC
                        print("[INFO] Quit requested.")
                        return

                    elif key == ord('1'):
                        cur = self.active_pairs[self.active_pair_idx] if self.active_pair_idx < len(self.active_pairs) else None
                        if cur is not None:
                            self._record_choice(cur[0])

                    elif key == ord('2'):
                        cur = self.active_pairs[self.active_pair_idx] if self.active_pair_idx < len(self.active_pairs) else None
                        if cur is not None:
                            self._record_choice(cur[1])
                    elif key == ord('3'):
                        cur = self.active_pairs[self.active_pair_idx] if self.active_pair_idx < len(self.active_pairs) else None
                        if cur is not None:
                            self._record_choice(BAD_BOTH)
                    elif key == ord('4'):
                        cur = self.active_pairs[self.active_pair_idx] if self.active_pair_idx < len(self.active_pairs) else None
                        if cur is not None:
                            self._record_choice(TIE_BOTH)
                    elif key == 83: # Right Arrow → finalize & go to next
                        cur = self.active_pairs[self.active_pair_idx] if self.active_pair_idx < len(self.active_pairs) else None
                        self._record_choice(cur[self.initial_choice])

                    elif key == 81:  # Left Arrow → go back one (no save)
                        print("[INFO] Back one frame.")
                        i = max(0, i - 1)
                        break
                    else:
                        continue

                    if self._finalize_and_advance:
                        self.log_frame()
                        i += 1
                        break

        finally:
            self._close_bag_doc()

if __name__ == "__main__":

    # ===========================
    # Configs
    # ===========================

    bag_dir = "/media/beast-gamma/Media/Datasets/SCAND/annt"   # Point to path with rosbags being annotated for the day
    expert_action_annotation_dir = "/media/beast-gamma/Media/Datasets/SCAND/ActionAnnotations"
    annotations_root = "./Annotations"
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
        annotator = TemporalAnnotator(bp, calib_path, topic_json_path, annotations_root, expert_action_annotation_dir, lookahead, num_keypoints, max_deviation)
        annotator.process_bag(undersampling_factor)

    print(f"\n[DONE] Annotations written to {annotator.output_path}")
