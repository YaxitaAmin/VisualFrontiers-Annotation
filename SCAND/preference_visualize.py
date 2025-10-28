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
    project_clip, make_corridor_polygon_from_cam_lines
from traj_utils import solve_arc_from_point, arc_to_traj, make_offset_paths, create_yaws_from_path
from utils import get_topics_from_bag

# Colors (BGR)
COLOR_PATH = (0, 0, 255)    # RED
COLOR_LAST = (0, 165, 255)  # ORANGE
COLOR_CLICK = (255, 0, 0)   # BLUE

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

class PathItem:
    path_points: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    polygon: np.ndarray

# ===========================
# Main Annotator
# ===========================
class PreferenceAnnotator:
    def __init__(self, bag_path, calib_path, topics_path, annotations_root, lookahead=5, num_keypoints=5, max_deviation=1.5):
        self.bag_path = bag_path
        self.bag_name = Path(bag_path).name
        self.needs_correction = False
        stem = Path(self.bag_name).stem
        self.output_path = os.path.join(annotations_root, f"{stem}.json")

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
        self.odom_topic = topics.get(mode).get("odom")
        self.robot_width = topics.get(mode).get("width")
        self.lookahead = lookahead
        self.keypoint_res = lookahead/num_keypoints

        print(f"[INFO] Using image topic: {self.image_topic}, control topic: {self.odom_topic}, robot width: {self.robot_width}")
 
        self.bridge = CvBridge()

        self.window = "SCAND Preference Annotator"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        self.current_img = None
        self.current_img_show = None        

        self.current_target_base = None   # (x,y,0) in base_link
        self.current_r_theta_vw = None    # (r, theta, v, w)

        self.writer = None

        self.last_selection_record = None  # (r,θ,v,ω,thick)
        self.last_click_uv = None
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

    def _open_bag_doc(self):
        self.bag_doc = {
            "bag": self.bag_name,
            "image_topic": self.image_topic,
            "annotations_by_stamp": {}
        }

    def _close_bag_doc(self):
        if self.bag_doc is not None:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.bag_doc, f, ensure_ascii=False, indent=2)
            self.bag_doc = None

    def draw(self):
        if self.current_img is None:
            return
        img = self.current_img.copy()
        img_h, img_w = img.shape[:2]
        # Draw centerline in red
        if self.path is not None:
            # Build corridor (in base frame)
            left_b, right_b, poly_b = make_corridor_polygon(self.path, self.yaws, self.robot_width)

            # Centerline & edges: project → clip bottom → optional smooth first segment
            points_2d = project_clip(self.path, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)

            # print(points_2d.shape)
            left_2d   = project_clip(left_b, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            right_2d  = project_clip(right_b, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)

            points_2d_p1 = project_clip(self.comparison_paths[0], self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            points_2d_p2 = project_clip(self.comparison_paths[1], self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True)
            # Corridor fill polygon: you can either rebuild from clipped edges or just project+clip the polygon
            # (If poly_b is already a stitched left+right, clip it similarly)
            poly_2d   = make_corridor_polygon_from_cam_lines(left_2d, right_2d)


            draw_polyline(img, points_2d, 2, COLOR_PATH)
            draw_polyline(img, points_2d_p1, 2, (0,255,0))  # GREEN
            draw_polyline(img, points_2d_p2, 2, (255,0,255))  # PURPLE
            draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.35, fill_color=COLOR_PATH, edge_color=COLOR_PATH, edge_thickness=2)
            
        self.current_img_show = img
        cv2.imshow(self.window, self.current_img_show)

    def log_frame(self):
        if self.bag_doc is None:
            raise RuntimeError("bag doc not open")

        u, v = self.last_click_uv
        r, theta, _, _, _ = self.last_selection_record            

        if self.frame_stamp is None:
            return  # nothing to log

        stamp_key = str(self.frame_stamp)
        # print(stamp_key)
        self.bag_doc["annotations_by_stamp"][stamp_key] = {
            "frame_idx": int(self.frame_idx),
            "click": {"u": u, "v": v},
            "arc": {"r": r, "theta": theta}, 
            "robot_width": self.robot_width
        }

        # clear per-frame transient state
        self.current_target_base = None
        self.current_r_theta_vw = None

    def log_stop(self):
        if self.bag_doc is None or self.frame_stamp is None:
            return

        stamp_key = str(self.frame_stamp)  # keep same format you use in log_frame

        self.bag_doc["annotations_by_stamp"][stamp_key] = {
            "frame_idx": int(self.frame_idx),
            "stop": True,
            "click": None,                     # no pixel click
            "goal_base": {"x": 0.0, "y": 0.0, "z": 0.0},
            "arc": {"r": 0.0, "theta": 0.0}
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
        
        yaw = msg.pose.pose.orientation.z
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        w = msg.twist.twist.angular.z

        return pos,v,w, rotation_matrix, yaw

    def process_image(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        return cv_img
    
    def create_path_item(self, path_points: np.ndarray, yaws: np.ndarray) -> PathItem:

        if yaws is None:
            yaws = create_yaws_from_path(path_points)

        left_b, right_b, poly_b = make_corridor_polygon(path_points, yaws, self.robot_width)
        path_item = PathItem()

        path_item.path_points = path_points
        path_item.left_boundary = left_b
        path_item.right_boundary = right_b
        path_item.polygon = poly_b

        return path_item

    def compute_comparison_paths(self):

        offset = np.random.uniform(self.robot_width/2, self.max_deviation)
        if self.cum_dists[-1] == 0:
            return np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([[0, 0, 0]])

        offset_ratios = self.cum_dists / self.cum_dists[-1]
        offsets = offset * (offset_ratios)

        left_offset_path, right_offset_path = make_offset_paths(self.path, self.yaws, offsets)

        #Hann conv offsets
        offset = np.random.uniform(self.robot_width/2, self.robot_width)
        base = 0.5 * (1.0 - np.cos(2.0 * np.pi * offset_ratios))
        gamma = 1.0
        w = base ** gamma
        conv_offsets = offset * w

        # print(conv_offsets)
        left_conv_path, right_conv_path = make_offset_paths(self.path, self.yaws, conv_offsets)

        return left_offset_path, right_offset_path, left_conv_path, right_conv_path
    
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

        self.lookahead = self.dynamic_lookahead(v, w)
        
        path = [current_pos]
        yaws = [current_yaw]
        cum_dists = [0]
        while True:

            # print(arc_length, self.lookahead)
            if frame_pointer > len(self.frames) - 1 :
                # print("Hmm")
                break

            pos_w = self.frames[frame_pointer].position
            yaw_t = self.frames[frame_pointer].yaw
            diff = (pos_w - path[-1])         #robots frame pos

            distance = np.linalg.norm(diff)
            arc_length += distance

            # print(distance)
            if arc_length > self.lookahead:
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
                      L_min: float = 1.0,        # never smaller than this [m]
                      L_max: float = 8.0,       # never larger than this [m]
                      kappa_gain: float = 2.0,   # how much to shrink on curves
                      eps: float = 1e-3) -> float:
        """Return meters of lookahead based on speed & curvature."""
        v = max(0.0, v)                              # ignore reverse for now
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
        
        count = 0
        with rosbag.Bag(self.bag_path, "r") as bag:
            pos_defined = False
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic, self.odom_topic])):
                
                if topic == self.odom_topic:
                    pos, v, w, rot, yaw = self.process_odom(msg)
                    pos_defined = True
                elif topic == self.image_topic:
                    cv_img = self.process_image(msg)

                    if pos_defined and i%undersampling_factor == 0:
                        self.frames.append(FrameItem(idx=count, stamp=t, img=cv_img, position=pos, velocity=v, omega=w, rotation=rot, yaw = yaw))   
                        count+=1        

        if not self.frames:
            print("[WARN] No frames after undersampling.")
            return

        i = 0
        try:
            while 0 <= i < len(self.frames):
                fr = self.frames[i]
                self.frame_idx = fr.idx
                self.frame_stamp = fr.stamp
                self.current_img = fr.img
                
                self.path, self.yaws, self.cum_dists = self.compute_path()
                self.paths.append(self.create_path_item(self.path, self.yaws))
                # print(self.path)
                left_offset_path, right_offset_path, left_conv_path, right_conv_path = self.compute_comparison_paths()
                self.comparison_paths = [left_offset_path, right_offset_path, left_conv_path, right_conv_path]

                # for p in self.comparison_paths:
                #     self.paths.append(self.create_path_item(p, yaws=None))


                self.draw()
                key = cv2.waitKey(0) & 0xFF

                if key in (ord('q'), 27):   # q or ESC
                    print("[INFO] Quit requested.")
                    return

                elif key == 83:  # Right Arrow → save (using last_*) then next
                    # self.log_frame()
                    i += 1

                elif key == 81:  # Left Arrow → go back one (no save)
                    print("[INFO] Back one frame.")
                    i = max(0, i - 1)
                else:
                    continue

        finally:
            self._close_bag_doc()

if __name__ == "__main__":

    # ===========================
    # Configs
    # ===========================

    bag_dir = "/media/beast-gamma/Media/Datasets/SCAND/annt"   # Point to path with rosbags being annotated for the day
    annotations_root = "./Annotations"
    calib_path = "./tf.json"
    skip_json_path = "./bags_to_skip.json"
    topic_json_path = "./topics_for_project.json"

    fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
    T_horizon = 2.0      # Path generation options
    num_t_samples = 1000
    robot_width_min = 0.35
    robot_width_max = 0.7
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
        
        annotator = PreferenceAnnotator(bp, calib_path, topic_json_path, annotations_root, lookahead, num_keypoints, max_deviation)
        annotator.process_bag(undersampling_factor)

    print(f"\n[DONE] Annotations written to {annotator.output_path}")
