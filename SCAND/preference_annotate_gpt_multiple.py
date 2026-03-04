#!/usr/bin/env python3

import os
import glob

from pathlib import Path
from typing import Optional, Tuple
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import rosbag

from vis_utils import draw_polyline, draw_corridor, project_clip, make_corridor_polygon_from_cam_lines, clean_2d
from preference_annotate import TemporalAnnotator, FrameItem, PathItem

from openai import OpenAI
import base64

from concurrent.futures import ThreadPoolExecutor, as_completed

# Colors (BGR) — swap out red/green
COLOR_PATH_ONE = (178, 114, 0)   # Blue  #0072B2
COLOR_PATH_TWO = (0, 159, 230)   # Orange #E69F00

# COLOR_PATH_ONE = (0, 0, 255)   # Red
# COLOR_PATH_TWO = (0, 0, 255)   # Red

EDGE_COLOR_ONE = (130, 80, 0)
EDGE_COLOR_TWO = (0, 120, 180)

# EDGE_COLOR_ONE = (0, 0, 255)
# EDGE_COLOR_TWO = (0, 0, 255)

FILL_ALPHA = 0.35

BAD_BOTH = 500
TIE_BOTH = 404

PROMPT_SYSTEM = """You are an expert human-preference annotator for mobile robot navigation.
You will see an image with two overlaid paths.
Each path has an associated number on the image: 1 on BLUE (left), 2 on ORANGE (right).
Color, number and side are only identifiers — do NOT prefer any color or side by default.

Prioritize, in order:
1. Safety — avoid pedestrians, obstacles, or collisions; aim ≥0.6 m clearance.
2. Viability — stay on walkable regions; avoid off-limits or unreachable regions.
3. Social compliance — do not cut through groups or disrupt natural flow.

Center bias:
- Prefer staying centered in the navigable area and navigating along the current field of view.
- Do NOT deviate from the middle unless an obstacle is inside the path boundaries NOW
  or is likely to ENTER the path within ~2 s based on its motion.
- Ignore obstacles clearly OUTSIDE the path boundaries or FAR AHEAD (>8 m) unless moving into it soon.

Obstacle-heading rule:
- DOWN-RANK any path that trends directly toward an obstacle (e.g., person, wall, barrier)
  that lies inside the path boundaries within the near range or at the path's end.

Decision rule:
- Pick 1 if the BLUE path (which is marked as 1) is better, 2 if the ORANGE path (which is marked as 2) is better.
- 500 (both bad): both paths clearly lead to collisions, obstacles, or off-limits areas.
- 404 (no preference): one or both paths are not visible or effectively identical.

Return ONLY this JSON:
{
  "choice": 1 | 2 | 500 | 404,
  "reason": "<=10 words>",
  "scores": { "safety": 0-10, "clearance": 0-10, "walkable": 0-10 }
}
Do not include anything outside the JSON.

For example if path 1 is heading towards an obstacle and path 2 is heading towards a clear area you would respond with: {"choice": 2, "reason": "Path 2 is clear", "scores": {"safety": 8, "clearance": 9, "walkable": 10}}
For example if both paths are heading towards obstacles you would respond with: {"choice": 500, "reason": "Both paths unsafe", "scores": {"safety": 2, "clearance": 3, "walkable": 4}}
For example if both paths are identical you would respond with: {"choice": 404, "reason": "Paths identical", "scores": {"safety": 7, "clearance": 7, "walkable": 7}}
"""

# PROMPT_SYSTEM = """You are an expert human-preference annotator for mobile robot navigation.
# You will see an image with two overlaid paths.
# Each path has an associated number on the image: 1 or 2. Your task is to pick the better path.

# Prioritize, in order:
# 1. Safety — avoid pedestrians, obstacles, or collisions; aim ≥0.6 m clearance.
# 2. Viability — stay on walkable regions; avoid off-limits or unreachable regions.
# 3. Social compliance — do not cut through groups or disrupt natural flow.

# Center bias:
# - Prefer staying centered in the navigable area and navigating along the current field of view.
# - Do NOT deviate from the middle unless an obstacle is inside the path boundaries NOW
#   or is likely to ENTER the path within ~2 s based on its motion.
# - Ignore obstacles clearly OUTSIDE the path boundaries or FAR AHEAD (>8 m) unless moving into it soon.

# Obstacle-heading rule:
# - DOWN-RANK any path that trends directly toward an obstacle (e.g., person, wall, barrier)
#   that lies inside the path boundaries within the near range or at the path's end.

# Decision rule:
# - 500 (both bad): both paths clearly lead to collisions, obstacles, or off-limits areas.
# - 404 (no preference): one or both paths are not visible or effectively identical.

# Return ONLY this JSON:
# {
#   "choice": 1 | 2 | 500 | 404,
#   "reason": "<=10 words>",
#   "scores": { "safety": 0-10, "clearance": 0-10, "walkable": 0-10 }
# }
# Do not include anything outside the JSON.


# """

class GPTModelClient:
    def __init__(self, model_name: str = "gpt-5-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. Please set it as an environment variable.")
        self.client = OpenAI(api_key=api_key)
        self.model = model_name

    def choose(self, img_bgr: np.ndarray, frame_idx: int, robot_width: float) -> dict:
        """Returns parsed dict: {'choice': int, 'reason': str, 'scores': {...}}"""
        data_url = self._bgr_to_dataurl(img_bgr)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": self.build_user_prompt(frame_idx)},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ]
        )
        text = resp.choices[0].message.content.strip()
        # be robust to accidental fencing
        text = text[text.find("{") : text.rfind("}") + 1]
        return json.loads(text)
    
    def _bgr_to_dataurl(self, img_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("PNG encode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def build_user_prompt(self, frame_idx: int) -> str:
        return (
            "Task: Compare two candidate paths overlaid on the image and return the JSON.\n"
            "Use the rules from the system message to pick the better path (1 or 2).\n"
            f'Frame info: {{"frame_idx": {frame_idx}}}'
        )

class TemporalAnnotatorGPT(TemporalAnnotator):
    def __init__(self, bag_path, calib_path, topics_path, annotations_root, expert_action_annotation_dir, lookahead=5, num_keypoints=5, max_deviation=1.5, preview_mode=False, preview_stride=10, preview_wait_ms=1):
        super().__init__(bag_path, calib_path, topics_path, annotations_root, expert_action_annotation_dir, lookahead, num_keypoints, max_deviation)
        self.gpt = GPTModelClient()
        self.preview_mode = preview_mode
        self.preview_stride = max(1, preview_stride)
        self.preview_wait_ms = max(1, preview_wait_ms)

    def _get_timestamps_from_expert_annotations(self):
        timestamps = []
        for key in self.action_annotations.get("annotations_by_stamp", {}).keys():
            timestamps.append(int(key))
        return timestamps

    def _pair_left_right_indices(self, pair) -> Tuple[int, int]:
        """Return (left_idx, right_idx) for the two path indices in `pair`,
        based on image-plane x positions of projected polylines.
        Falls back to world-frame lateral check if projection fails."""
        i, j = pair
        img_h, img_w = self.current_img.shape[:2]

        def median_x(pitem: PathItem):
            pts_2d = clean_2d(project_clip(
                pitem.path_points, self.T_cam_from_base, self.K, self.dist,
                img_h, img_w, smooth_first=True
            ), img_w, img_h)
            if pts_2d.shape[0] == 0:
                return None
            # use median x to be robust to partial visibility
            return float(np.median(pts_2d[:, 0]))

        xi = median_x(self.paths[i])
        xj = median_x(self.paths[j])

        if xi is not None and xj is not None:
            return (i, j) if xi <= xj else (j, i)

        # Fallback: world-frame lateral (y) of last point in robot/base frame
        pi = self.paths[i].path_points[-1]
        pj = self.paths[j].path_points[-1]
        return (i, j) if pi[1] <= pj[1] else (j, i)
    
    def _draw_circle_badge(self, img, center_xy, text, radius=16,
                       bg=(30,30,30), fg=(255,255,255), ring=(255,255,255)):
        """Draw a circular badge with a thin white ring and centered text."""
        x, y = center_xy
        cv2.circle(img, (x, y), radius+2, ring, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, bg, thickness=-1, lineType=cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.6, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx, ty = int(x - tw/2), int(y + th/2 - 2)
        cv2.putText(img, text, (tx, ty), font, scale, fg, thickness, cv2.LINE_AA)


    def render_pair_image(self, pair):
        if self.current_img is None:
            return None
        img = self.current_img.copy()
        img_h, img_w = img.shape[:2]

        left_idx, right_idx = self._pair_left_right_indices(pair)

        def draw_one(pitem, color, edge_color):
            points_2d = clean_2d(project_clip(pitem.path_points, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            left_2d   = clean_2d(project_clip(pitem.left_boundary,  self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            right_2d  = clean_2d(project_clip(pitem.right_boundary, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
            poly_2d   = make_corridor_polygon_from_cam_lines(left_2d, right_2d)
            draw_polyline(img, points_2d, 2, color)
            draw_corridor(img, poly_2d, left_2d, right_2d,
                        fill_alpha=FILL_ALPHA, fill_color=color,
                        edge_color=edge_color, edge_thickness=2)
            
        def _end_xy(pitem):
            pts_2d = clean_2d(project_clip(
                pitem.path_points, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True
            ), img_w, img_h)
            if pts_2d.shape[0] == 0:
                return None
            # use last visible point; nudge upward a bit to avoid being on edge
            x, y = pts_2d[-1, 0], max(0, pts_2d[-1, 1] - 12)
            return (int(x), int(y))

        # BLUE(1) must be the LEFT one; ORANGE(2) the RIGHT one
        draw_one(self.paths[left_idx],  COLOR_PATH_ONE, EDGE_COLOR_ONE)
        draw_one(self.paths[right_idx], COLOR_PATH_TWO, EDGE_COLOR_TWO)

        # BLUE/left badge "1"
        pL = _end_xy(self.paths[left_idx])
        if pL is not None:
            self._draw_circle_badge(img, pL, "1", fg=(255,255,255), bg=(40,80,160))

        # ORANGE/right badge "2"
        pR = _end_xy(self.paths[right_idx])
        if pR is not None:
            self._draw_circle_badge(img, pR, "2", fg=(255,255,255), bg=(20,120,200))

        return img

    def _record_choice(self, pair, chosen_idx: int, meta: Optional[dict] = None):
        if self.active_pair_idx < 0 or self.active_pair_idx >= len(self.active_pairs):
            return

        entry = {
            "pair": [int(pair[0]), int(pair[1])],
            "choice": int(chosen_idx),
            "meaning": "both_bad" if chosen_idx == BAD_BOTH else ("no_preference" if chosen_idx == TIE_BOTH else "picked_index")
        }
        if meta:
            entry["model_meta"] = meta  # {"reason": "...", "scores": {...}}
        self.prefs_this_frame.append(entry)
        self.active_pair_idx += 1
        if self.active_pair_idx == len(self.active_pairs):
            self._finalize_and_advance = True
    
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
                print(f"[INFO] Processing frame {i+1}/{len(self.frames)} (idx={self.frames[i].idx})")
                fr = self.frames[i]
                self.frame_idx = fr.idx
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

                pairs = self.active_pairs[:]
                futures = []
                with ThreadPoolExecutor(max_workers=6) as ex:
                    for pair in pairs:
                        img_pair = self.render_pair_image(pair)
                        futures.append((pair, ex.submit(self.gpt.choose, img_pair, self.frame_idx, self.robot_width)))

                for pair, fut in futures:
                    model_out = fut.result()
                    raw_choice = int(model_out.get("choice", TIE_BOTH))
                    meta = {"reason": model_out.get("reason",""), "scores": model_out.get("scores",{})}
                    left_idx, right_idx = self._pair_left_right_indices(pair)

                    if raw_choice == 1:          # BLUE == LEFT
                        self._record_choice(pair, left_idx, meta)
                    elif raw_choice == 2:        # ORANGE == RIGHT
                        self._record_choice(pair, right_idx, meta)
                    elif raw_choice == 500:
                        self._record_choice(pair, BAD_BOTH, meta)
                    else:
                        self._record_choice(pair, TIE_BOTH, meta)

                self.log_frame()
                i += 1

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
        annotator = TemporalAnnotatorGPT(bp, calib_path, topic_json_path, annotations_root, expert_action_annotation_dir, lookahead, num_keypoints, max_deviation)
        annotator.process_bag(undersampling_factor)

    print(f"\n[DONE] Annotations written to {annotator.output_path}")
