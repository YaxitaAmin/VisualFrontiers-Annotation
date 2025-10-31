from utils import get_existing_bags, get_topics_and_info_from_bag, get_topics_from_bag
from vis_utils import load_calibration

import rosbag
import json
import numpy as np
import os
import math

from cv_bridge import CvBridge
import cv2

from vis_utils import transform_points, project_points_cam
from utils import ensure_csv_has_header, append_csv_row

from scipy.spatial.transform import Rotation as R

fx, fy, cx, cy = 640, 637, 640, 360

rosbag_root = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/"
skip_json_path = "./bags_to_skip.json"
topics_for_project_path = "./topics_for_project.json"
calib_path = "./tf.json"

T_horizon = 2.0      # Path generation options
out_csv_path = "./oov_counts.csv"

visualize = False

def count_oov_action():

    ensure_csv_has_header(out_csv_path)

    with open(skip_json_path, 'r') as f:
        bags_to_skip = json.load(f)
    
    with open(topics_for_project_path, 'r') as f:
        topics_for_project = json.load(f)
    
    existing_bags = get_existing_bags(rosbag_root)
    bag_files = sorted([f for f in existing_bags if not bags_to_skip.get(f, False)])

    oov_counts = {}
    bridge = CvBridge()

    for bag_file in bag_files:

        if bags_to_skip.get(bag_file, False):
            print(f"[INFO] Skipping {bag_file}")
            continue

        bag_path = os.path.join(rosbag_root, bag_file)
        print(f"[INFO] Processing {bag_file}")

        needs_correction = False
        if "Jackal" in bag_file:
            cam_topic = topics_for_project.get("jackal").get("camera")
            control_topic = topics_for_project.get("jackal").get("odom")
            width = topics_for_project.get("jackal").get("width")
            K, dist, T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="jackal")

        elif "Spot" in bag_file:
            cam_topic = topics_for_project.get("spot").get("camera")
            control_topic = topics_for_project.get("spot").get("odom")
            width = topics_for_project.get("spot").get("width")
            K, dist, T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="spot")
            needs_correction = True

        else:
            print(f"[WARN] Unknown robot in bag name: {bag_file}")
            continue
        
        T_cam_from_base = np.linalg.inv(T_base_from_cam)
        oov_count = 0
        total_count = 0

        try:
            with rosbag.Bag(bag_path, 'r') as bag:

                current_img = None
                v = None
                w = None
                for topic, msg, t in bag.read_messages(topics=[cam_topic, control_topic]):

                    if topic == control_topic:
                        if needs_correction:
                            current_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                            quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
                            rotation_matrix = R.from_quat(quaternion).as_matrix()
                            velocity_robot_frame = np.linalg.inv(rotation_matrix) @ current_vel

                            v = velocity_robot_frame[0]
                        else:
                            v = msg.twist.twist.linear.x

                        w = msg.twist.twist.angular.z                    
                        within_bounds, point = check_within_bounds(v, w, T_cam_from_base, K, dist)

                        total_count += 1
                        if not within_bounds:
                            oov_count += 1

                    elif topic == cam_topic:
                        current_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

                    if visualize:
                        if current_img is not None and v is not None and w is not None:
                            draw(current_img, point, v, w)

                    
        except Exception as e:
            print(f"[ERROR] Failed processing {bag_file}: {e}")

        # Write one line for this bag immediately
        append_csv_row(out_csv_path, bag_file, oov_count, total_count)
        print(f"[INFO] {bag_file}: OOV={oov_count}, TOTAL={total_count}")

def check_within_bounds(v, w, T_cam_from_base, K, dist):

    theta = w * T_horizon
    
    x_t = (v/w) * math.sin(theta) if abs(w) > 1e-3 else v * T_horizon 
    y_t = (v/w) * (1 - math.cos(theta)) if abs(w) > 1e-3 else 0.0 

    g_r = np.array([[x_t, y_t, 0.0]])

    g_c = transform_points(T_cam_from_base, g_r)
    g_img = project_points_cam(K, dist, g_c)

    # Valid if projection returned at least one finite pixel

    px, py = float(g_img[0, 0]), float(g_img[0, 1])
    pxi, pyi = int(round(px)), int(round(py))

    within = False
    if 0 <= pxi < 1280 and 0 <= pyi < 720:
        within = True
    
    return within, g_img

def draw(img, point, v, w, window_name="Goal Point Visualization"):
    if img is None:
        return

    # Prepare visualization frame
    vis = img.copy()
    h, w_img = vis.shape[:2]

    if point is not None and np.all(np.isfinite(point)):
        px, py = float(point[0, 0]), float(point[0, 1])
        pxi, pyi = int(round(px)), int(round(py))

        if 0 <= pxi < w_img and 0 <= pyi < h:
            cv2.circle(vis, (pxi, pyi), 6, (0, 255, 0), -1)
            cv2.putText(vis, "goal", (pxi + 8, pyi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "goal off-image",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"u :{pxi} , v: {pyi}",
                        (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, "no goal projection",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(vis, f"v={v:.2f} m/s   w={w:.2f} rad/s",
                (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Press RIGHT ARROW to advance (q/ESC to skip)",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception:
        pass

    cv2.imshow(window_name, vis)

    while True:
        k = cv2.waitKey(0)
        if k in (83, 2555904):
            break
        if k in (ord('q'), 27):
            break

if __name__ == "__main__":
    count_oov_action()