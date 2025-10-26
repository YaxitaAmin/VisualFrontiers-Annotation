import os
import rosbag
from cv_bridge import CvBridge
import cv2
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import math

from vis_utils import point_to_traj, make_corridor_polygon, draw_polyline, draw_corridor, transform_points, \
    project_points_cam, load_calibration, camray_to_ground_in_base
from traj_utils import solve_arc_from_point

bagfile_root = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/"
annotations_root = "./Annotations"
fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
T_horizon = 2.0      # Path generation options
num_t_samples = 1000
calib_path = "./tf.json"

COLOR_PATH = (0, 0, 255) #red
COLOR_CLICK = (255, 0, 0) #green

@dataclass
class FrameItem:
    idx: int
    stamp: object   # rospy.Time
    img: np.ndarray
    r: float
    theta: float
    sub_goal : list   # [x, y, z] in base_link
    width_m : float
    u : float
    v : float
    stop : bool

def draw(frame_item: FrameItem, K: np.ndarray, dist, T_cam_from_base: np.ndarray, window: str = "SCAND Verification"):

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    current_img = frame_item.img
    x_base, y_base, z_base = frame_item.sub_goal
    width_m = frame_item.width_m
    r, theta = frame_item.r, frame_item.theta
    u, v = frame_item.u, frame_item.v
    stop = frame_item.stop

    if not stop: 

        r, theta = solve_arc_from_point(x_base, y_base)
        # print(x_base, y_base)
        path_points, v_x, w, t_samples, theta_samples = point_to_traj(r, theta, T_horizon, num_t_samples, x_base, y_base)
        left_b, right_b, poly_b = make_corridor_polygon(path_points, theta_samples, width_m, bridge_pts=20)

        # print(path_points[-10:, :])

        # 4) Transform to camera and project
        traj_c = transform_points(T_cam_from_base, path_points)
        left_c = transform_points(T_cam_from_base, left_b)
        right_c= transform_points(T_cam_from_base, right_b)
        poly_c = transform_points(T_cam_from_base, poly_b)

        # print(traj_c[-10:, :])

        ctr_2d  = project_points_cam(K, dist, traj_c)
        left_2d = project_points_cam(K, dist, left_c)
        right_2d= project_points_cam(K, dist, right_c)
        poly_2d = project_points_cam(K, dist, poly_c)
        # Project to image

        # print(ctr_2d[-10:, :])

        # print("\n")
        # print("Next")
        # print("\n")

        if current_img is None:
            return
        img = current_img.copy()


        draw_polyline(img, ctr_2d, 2, COLOR_PATH)
        draw_corridor(img, poly_2d, left_2d, right_2d, 
                    fill_alpha=0.35, fill_color=COLOR_PATH, edge_color=COLOR_PATH, edge_thickness=2)
        
        cv2.circle(img, (int(u), int(v)), 5, COLOR_CLICK, -1)
    else:
        ## transclucent red overlay for stop
        img = current_img.copy()
        overlay = img.copy()
        overlay[:] = (0,0,255)
        img[:] = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    cv2.imshow(window, img)

def verify_annotations():

    annotations_files = sorted([f for f in os.listdir(annotations_root) if f.endswith('.json')])
    bridge = CvBridge()

    for ann_file in annotations_files:

        if "Jackal" in ann_file:
            K, dist, T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="jackal")
            T_cam_from_base = np.linalg.inv(T_base_from_cam)
        elif "Spot" in ann_file:
            K, dist, T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="spot")
            T_cam_from_base = np.linalg.inv(T_base_from_cam)

        ann_path = os.path.join(annotations_root, ann_file)
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        bag_name = ann_data.get("bag", False)
        if not bag_name:
            print(f"[ERROR] No bag_name found in {ann_file}")
            continue

        img_topic = ann_data.get("image_topic", False)
        if not img_topic:
            print(f"[ERROR] No image_topic found in {ann_file}")
            continue

        bag_path = os.path.join(bagfile_root, bag_name)

        if not os.path.exists(bag_path):
            print(f"[ERROR] Bag file {bag_name} not found for annotation {ann_file}")
            continue

        print(f"[INFO] Verifying annotations in {ann_file} against bag {bag_name}")

        annotation_timstamps = ann_data.get("annotations_by_stamp")
        frame_items : list[FrameItem] = []

        with rosbag.Bag(bag_path, 'r') as bag:

            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=[img_topic])):
                annotation_item = annotation_timstamps.get(str(t), False)
                if not annotation_item:
                    continue

                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                sub_goal = annotation_item.get("goal_base")
                sub_goal = [sub_goal["x"], sub_goal["y"], sub_goal["z"]]
                width_m = annotation_item.get("robot_width", 0.5)

                stop = annotation_item.get("stop", False)

                if stop:
                    u, v = 0,0
                else:
                    u = annotation_item.get("click")["u"]
                    v = annotation_item.get("click")["v"]
                    P_b = camray_to_ground_in_base(u, v, K, T_base_from_cam)

                    sub_goal[0] = P_b[0]
                    sub_goal[1] = P_b[1]

                # print(sub_goal, P_b)
                r = annotation_item.get("arc")["r"]
                theta = annotation_item.get("arc")["theta"]
                
                frame_items.append(FrameItem(idx=idx, 
                                             stamp=t, 
                                             img=cv_img, 
                                             r=r, 
                                             theta=theta, 
                                             sub_goal=sub_goal, 
                                             width_m=width_m,
                                             u=u,
                                             v=v,
                                             stop=stop))
            i = 0
            while 0 <= i < len(frame_items):
                fr = frame_items[i]
                draw(fr, K, dist, T_cam_from_base)
                key = cv2.waitKey(0) & 0xFF

                if key == 83:  # Right Arrow → save (using last_*) then next
                    i += 1

                elif key == 81:  # Left Arrow → go back one (no save)
                    print("[INFO] Back one frame.")
                    i = max(0, i - 1)
                else:
                    continue
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_annotations()