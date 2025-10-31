#!/usr/bin/env python3
import os
import argparse
import csv
import math
from typing import Optional, Tuple, List

import cv2
import numpy as np

# ROS1 deps
import rosbag
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

def compute_pitch_deg(y_h: float, fy: float, cy: float) -> float:
    """
    θ ≈ atan((cy - y_h)/fy). Positive when horizon below cy (camera pitched down).
    Returns degrees.
    """
    theta = math.atan2((y_h-cy), fy)
    return math.degrees(theta)

class FrameRecord:
    __slots__ = ("stamp_ns", "img")
    def __init__(self, stamp_ns: int, img: np.ndarray):
        self.stamp_ns = stamp_ns
        self.img = img

def read_images_from_bag(bag_path: str, image_topic: str, max_frames: Optional[int]=None) -> List[FrameRecord]:
    bridge = CvBridge()
    frames = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            mtype = getattr(msg, "_type", "")
            if not mtype == "sensor_msgs/CompressedImage":
                continue
            try:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                print(f"[WARN] cv_bridge conversion failed: {e}")
                continue
            stamp_ns = int(msg.header.stamp.to_nsec()) if msg.header and msg.header.stamp else int(t.to_nsec())
            frames.append(FrameRecord(stamp_ns, cv_img))
            if max_frames and len(frames) >= max_frames:
                break
    if not frames:
        print(f"[ERROR] No frames found on topic '{image_topic}' in {bag_path}")
    else:
        print(f"[INFO] Loaded {len(frames)} frames from {bag_path} ({image_topic})")
    return frames

class PitchAnnotator:
    def __init__(self, frames: List[FrameRecord], intrinsics: Optional[Tuple[float,float,float,float]], out_csv: str):
        self.frames = frames
        self.fx_fy_cx_cy = intrinsics
        self.out_csv = out_csv

        self.idx = 0
        self.h_line = 0  # y position of horizon line
        self.dragging = False

        self.window = "Pitch Annotator"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.window, self.on_mouse)
        cv2.createTrackbar("y", self.window, 0, max(1, self.curr_img().shape[0] - 1), self.on_trackbar)

        # Initialize line at image mid-height
        h = self.curr_img().shape[0]
        self.h_line = h // 2
        cv2.setTrackbarPos("y", self.window, self.h_line)

        self.records = []  # list of dicts

    def curr_img(self) -> np.ndarray:
        return self.frames[self.idx].img

    def on_trackbar(self, val: int):
        self.h_line = int(val)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.h_line = int(y)
            cv2.setTrackbarPos("y", self.window, self.h_line)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.h_line = int(np.clip(y, 0, self.curr_img().shape[0] - 1))
            cv2.setTrackbarPos("y", self.window, self.h_line)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def draw_overlay(self, img: np.ndarray) -> np.ndarray:
        vis = img.copy()
        h, w = vis.shape[:2]
        # horizon line
        cv2.line(vis, (0, self.h_line), (w-1, self.h_line), (0, 255, 255), 2, cv2.LINE_AA)

        msg = f"Frame {self.idx+1}/{len(self.frames)} | y={self.h_line}"
        if self.fx_fy_cx_cy:
            fx, fy, cx, cy = self.fx_fy_cx_cy
            pitch_deg = compute_pitch_deg(self.h_line, fy, cy)
            msg += f" | pitch≈{pitch_deg:+.2f}°"

        # HUD text
        cv2.rectangle(vis, (10, 10), (10+600, 10+50), (0,0,0), -1)
        cv2.putText(vis, msg, (18, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, "Keys: s=save+next | n=next | p=prev | q=quit", (18, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return vis

    def save_current(self):
        rec = {
            "index": self.idx,
            "stamp_ns": self.frames[self.idx].stamp_ns,
            "y_px": int(self.h_line),
            "pitch_rad": "",
            "pitch_deg": "",
        }
        if self.fx_fy_cx_cy:
            fx, fy, cx, cy = self.fx_fy_cx_cy
            pitch_deg = compute_pitch_deg(self.h_line, fy, cy)
            rec["pitch_deg"] = f"{pitch_deg:.6f}"
            rec["pitch_rad"] = f"{math.radians(pitch_deg):.6f}"
        self.records.append(rec)
        print(f"[SAVE] idx={rec['index']} stamp={rec['stamp_ns']} y={rec['y_px']} "
              f"{'(pitch_deg='+str(rec['pitch_deg'])+')' if rec['pitch_deg']!='' else ''}")

    def write_csv(self):
        if not self.records:
            print("[INFO] No annotations to save.")
            return
        header = ["index", "stamp_ns", "y_px", "pitch_rad", "pitch_deg"]
        tmp_path = self.out_csv + ".tmp"
        with open(tmp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in self.records:
                w.writerow(r)
        os.replace(tmp_path, self.out_csv)
        print(f"[OK] Wrote {len(self.records)} annotations to {self.out_csv}")

    def run(self):
        while True:
            img = self.curr_img()
            vis = self.draw_overlay(img)
            cv2.imshow(self.window, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_current()
                # advance
                if self.idx < len(self.frames) - 1:
                    self.idx += 1
                    # keep line at same relative position or reset? keep as-is
                    cv2.setTrackbarPos("y", self.window, int(np.clip(self.h_line, 0, self.curr_img().shape[0]-1)))
                else:
                    print("[INFO] Last frame reached.")
                    break
            elif key == ord('n'):
                if self.idx < len(self.frames) - 1:
                    self.idx += 1
                    cv2.setTrackbarPos("y", self.window, int(np.clip(self.h_line, 0, self.curr_img().shape[0]-1)))
                else:
                    print("[INFO] Last frame reached.")
                    break
            elif key == ord('p'):
                if self.idx > 0:
                    self.idx -= 1
                    cv2.setTrackbarPos("y", self.window, int(np.clip(self.h_line, 0, self.curr_img().shape[0]-1)))
                else:
                    print("[INFO] First frame reached.")
                    break
        self.write_csv()
        cv2.destroyAllWindows()

def main():
    bag_file = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/A_Jackal_GDC_GDC_Fri_Oct_29_11.bag"
    image_topic = "/camera/rgb/image_raw/compressed"
    fx, fy, cx, cy = 640, 637, 640, 360
    out = "pitch_annotations.csv"
    intrinsics = fx, fy, cx, cy
    
    print(f"[INFO] Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    
    frames = read_images_from_bag(bag_file, image_topic)
    if not frames:
        return

    annot = PitchAnnotator(frames, intrinsics, out)
    annot.run()

if __name__ == "__main__":
    main()
