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

from dataclasses import dataclass

# --- ROS1 imports ---
import rosbag
from cv_bridge import CvBridge

from vis_utils import camray_to_ground_in_base, transform_points, load_calibration, \
    make_corridor_polygon, draw_polyline, draw_corridor, project_points_cam
from traj_utils import solve_arc_from_point, arc_to_traj
# ===========================
# Configs
# ===========================

bag_dir = "/media/beast-gamma/Media/Datasets/SCAND/annt"   # Point to path with rosbags being annotated for the day
annotations_root = "./Annotations"
calib_path = "./tf.json"
skip_json_path = "./bags_to_skip.json"

fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
T_horizon = 2.0      # Path generation options
num_t_samples = 1000
robot_width_min = 0.35
robot_width_max = 0.7
undersampling_factor = 6

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
    
# ===========================
# Main Annotator
# ===========================
class Annotator:
    def __init__(self):
        self.K, self.dist, self.T_base_from_cam = None, None, None
        self.T_cam_from_base = None
        self.bridge = CvBridge()

        self.window = "SCAND Annotator"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)

        self.current_img = None
        self.current_img_show = None
        self.image_topic = None

        self.latest_pts2d = None
        self.latest_left2d  = None
        self.latest_right2d = None
        self.latest_poly2d  = None
        self.robot_width = None

        self.last_carry_pts2d = None
        self.last_carry_left2d  = None
        self.last_carry_right2d = None
        self.last_carry_poly2d  = None
        self.last_carry_robot_width = None

        self.current_click_uv = None
        self.current_target_base = None   # (x,y,0) in base_link
        self.current_r_theta_vw = None    # (r, theta, v, w)

        self.writer = None

        self.last_target_base = None
        self.last_selection_record = None  # (r,θ,v,ω,thick)
        self.last_click_uv = None
        self.frame_idx = -1
        self.bag_name = ""
        self.frame_stamp = None

        self.bag_doc = None
        self.frames : None
        self.output_path = None

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
    
    def _clear_last(self):
        self.last_click_uv = None
        self.last_target_base = None
        self.last_selection_record = None
        self.last_carry_pts2d = None
        self.last_carry_left2d = None
        self.last_carry_right2d = None
        self.last_carry_poly2d = None
        self.last_carry_robot_width = None

        self.current_click_uv = None
        self.current_target_base = None
        self.current_r_theta_vw = None
        self.latest_pts2d = None
        self.latest_poly2d = None
        self.latest_left2d = None
        self.latest_right2d = None

    def on_mouse(self, event, x, y, flags, userdata):
        if self.current_img is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_click_uv = (float(x), float(y))
            self.last_click_uv = self.current_click_uv

            P_b = camray_to_ground_in_base(x, y, self.K, self.T_base_from_cam)
            if P_b is None:
                print("[WARN] Ray did not hit ground plane (z=0) in front of camera.")
                return
            xb, yb = float(P_b[0]), float(P_b[1])
            r, theta = solve_arc_from_point(xb, yb)

            traj_b, vel, w, t_arr, theta_arr = arc_to_traj(r, theta, T_horizon, num_t_samples, xb, yb)
            robot_width = random.uniform(robot_width_min, robot_width_max)
            left_b, right_b, poly_b = make_corridor_polygon(traj_b, theta_arr, robot_width)

            # 4) Transform to camera and project
            traj_c = transform_points(self.T_cam_from_base, traj_b)
            left_c = transform_points(self.T_cam_from_base, left_b)
            right_c= transform_points(self.T_cam_from_base, right_b)
            poly_c = transform_points(self.T_cam_from_base, poly_b)

            ctr_2d  = project_points_cam(self.K, self.dist, traj_c)
            left_2d = project_points_cam(self.K, self.dist, left_c)
            right_2d= project_points_cam(self.K, self.dist, right_c)
            poly_2d = project_points_cam(self.K, self.dist, poly_c)


            # Keep your red centerline if desired:
            self.latest_pts2d = ctr_2d  # centerline

            # New: store corridor for redraw
            self.latest_left2d  = left_2d
            self.latest_right2d = right_2d
            self.latest_poly2d  = poly_2d
            self.robot_width = robot_width

            self.current_target_base = (xb, yb)
            self.last_target_base = self.current_target_base
            self.current_r_theta_vw = (r, theta, vel, w, robot_width)

            self.last_carry_pts2d = ctr_2d.copy()
            self.last_carry_left2d  = left_2d.copy()
            self.last_carry_right2d = right_2d.copy()
            self.last_carry_poly2d  = poly_2d.copy()
            self.last_carry_robot_width = robot_width
            self.last_selection_record = (r, theta, vel, w, robot_width)

            print(f"[INFO] Timestamp : {self.frame_stamp}, Click {(x,y)} → base ({xb:.3f},{yb:.3f}), r={r:.3f}, θ={np.rad2deg(theta):.3f}, v={vel:.3f}, ω={w:.3f}, ")
            self.redraw()

    def redraw(self):
        if self.current_img is None:
            return
        img = self.current_img.copy()

        # Draw carried-over corridor if no new latest
        if self.last_carry_pts2d is not None and self.latest_pts2d is None:
            draw_polyline(img, self.last_carry_pts2d, 2, COLOR_LAST)

        if self.last_carry_poly2d is not None and self.latest_poly2d is None:
            draw_corridor(img, self.last_carry_poly2d, self.last_carry_left2d, self.last_carry_right2d,
                        fill_alpha=0.35, fill_color=COLOR_LAST, edge_color=COLOR_LAST, edge_thickness=2)

        # Draw current corridor (translucent fill + solid edges)
        if self.latest_poly2d is not None:
            draw_corridor(img, self.latest_poly2d, self.latest_left2d, self.latest_right2d,
                        fill_alpha=0.35, fill_color=COLOR_PATH, edge_color=(0,0,200), edge_thickness=2)

        # Draw centerline in red
        if self.latest_pts2d is not None:
            draw_polyline(img, self.latest_pts2d, 2, COLOR_PATH)

        if self.current_click_uv is not None:
            cv2.circle(img, (int(self.current_click_uv[0]), int(self.current_click_uv[1])), 5, COLOR_CLICK, -1)

        self.current_img_show = img
        cv2.imshow(self.window, self.current_img_show)

    def log_frame(self):
        if self.bag_doc is None:
            raise RuntimeError("bag doc not open")
        if not (self.last_click_uv and self.last_target_base and self.last_selection_record):
            return
        
        u, v = self.last_click_uv
        xb, yb = self.last_target_base
        r, theta, _, _, _ = self.last_selection_record            

        if self.frame_stamp is None:
            return  # nothing to log

        stamp_key = str(self.frame_stamp)
        # print(stamp_key)
        self.bag_doc["annotations_by_stamp"][stamp_key] = {
            "frame_idx": int(self.frame_idx),
            "click": {"u": u, "v": v},
            "goal_base": {"x": xb, "y": yb, "z": 0.0},
            "arc": {"r": r, "theta": theta}, 
            "robot_width": self.last_carry_robot_width
        }

        # clear per-frame transient state
        self.current_click_uv = None
        self.current_target_base = None
        self.current_r_theta_vw = None
        self.latest_pts2d = None
        self.latest_poly2d = None
        self.latest_left2d = None
        self.latest_right2d = None

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

    def process_bag(self, bag_path: str):
        self.bag_name = Path(bag_path).name
        stem = Path(self.bag_name).stem
        self.output_path = os.path.join(annotations_root, f"{stem}.json")

        print(f"\n=== Processing {self.bag_name} ===")

        with rosbag.Bag(bag_path, "r") as bag:
            if "Jackal" in self.bag_name:
                self.image_topic = "/camera/rgb/image_raw/compressed"
                self.K, self.dist, self.T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="jackal")
                self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
            elif "Spot" in self.bag_name:
                self.image_topic = "/image_raw/compressed"
                self.K, self.dist, self.T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="spot")
                self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
            print(f"[INFO] Using image topic: {self.image_topic}")

            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic])):

                if i % undersampling_factor != 0:
                    continue

                self.frame_idx = i
                self.frame_stamp = t

                # Convert ROS Image -> BGR
                cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self.frames.append(FrameItem(idx=i, stamp=t, img=cv_img))

        if not self.frames:
            print("[WARN] No frames after undersampling.")
            return

        self._open_bag_doc()
        i = 0
        try:
            while 0 <= i < len(self.frames):
                fr = self.frames[i]
                self.frame_idx = fr.idx
                self.frame_stamp = fr.stamp
                self.current_img = fr.img

                # clear “current” state (but keep last_* to allow carry-over)
                self.current_click_uv = None
                self.current_target_base = None
                self.current_r_theta_vw = None
                self.latest_pts2d = None
                self.latest_poly2d = None
                self.latest_left2d = None
                self.latest_right2d = None

                self.redraw()
                key = cv2.waitKey(0) & 0xFF

                if key in (ord('q'), 27):   # q or ESC
                    print("[INFO] Quit requested.")
                    return

                elif key == 83:  # Right Arrow → save (using last_*) then next
                    self.log_frame()
                    i += 1

                elif key == 81:  # Left Arrow → go back one (no save)
                    print("[INFO] Back one frame.")
                    i = max(0, i - 1)
                elif key == ord('0'):  # STOP action
                    self.log_stop()
                    self._clear_last()
                    self.redraw()
                    continue 
                else:
                    continue

        finally:
            self._close_bag_doc()

    def run(self):
        bag_files = sorted(glob.glob(os.path.join(bag_dir, "*.bag")))

        with open(skip_json_path, 'r') as f:
            bags_to_skip = json.load(f)
        if not bag_files:
            print(f"[ERROR] No .bag files found in {bag_dir}")
            return
        for bp in bag_files:
            print(os.path.basename(bp))
            if bags_to_skip.get(os.path.basename(bp), False):
                print(f"[INFO] Skipping {bp}")
                continue
            self.frames : list[FrameItem] = []
            self.process_bag(bp)
        print(f"\n[DONE] Annotations written to {self.output_path}")

if __name__ == "__main__":
    Annotator().run()
