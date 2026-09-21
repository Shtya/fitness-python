"""MediaPipe Pose + silhouette body measurement estimator.

Never invents values. Returns null when landmarks, silhouette, or
anthropometric plausibility checks fail.
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
)
from PIL import Image, ImageOps

LM = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Min / max as a fraction of standing height. Outside this = unreliable.
PLAUSIBLE = {
    "shoulderWidth": (0.18, 0.34),
    "chest": (0.38, 0.72),
    "waist": (0.30, 0.68),
    "hips": (0.38, 0.74),
    "upperArm": (0.11, 0.26),
    "thigh": (0.20, 0.42),
    "inseam": (0.38, 0.56),
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "pose_landmarker_heavy.task")

_landmarker: Optional[PoseLandmarker] = None


@dataclass
class PoseView:
    image_rgb: np.ndarray
    mask: np.ndarray
    landmarks: list
    width: int
    height: int
    scale_cm_per_px: float
    body_top: int
    body_bottom: int


def ensure_model() -> str:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return MODEL_PATH
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def get_landmarker() -> PoseLandmarker:
    global _landmarker
    if _landmarker is None:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
        )
        _landmarker = PoseLandmarker.create_from_options(options)
    return _landmarker


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img)
    h, w = arr.shape[:2]
    max_side = 1280
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return arr


def vis(lm, idx: int) -> float:
    item = lm[idx]
    value = getattr(item, "visibility", None)
    if value is None:
        value = getattr(item, "presence", None)
    if value is None:
        return 1.0
    return float(value)


def xy(lm, idx: int, width: int, height: int) -> tuple[float, float]:
    item = lm[idx]
    return float(item.x) * width, float(item.y) * height


def ellipse_circumference(width_cm: float, depth_cm: float) -> Optional[float]:
    a = width_cm / 2.0
    b = depth_cm / 2.0
    if a <= 0 or b <= 0:
        return None
    # Ramanujan approximation
    value = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return value if math.isfinite(value) and value > 0 else None


def mask_width_at(mask: np.ndarray, y: int, search: int = 6) -> Optional[int]:
    h, w = mask.shape
    y = int(np.clip(y, 0, h - 1))
    best = None
    for dy in range(-search, search + 1):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        cols = np.flatnonzero(mask[yy])
        if cols.size < 4:
            continue
        width = int(cols[-1] - cols[0] + 1)
        if best is None or width > best:
            best = width
    return best


def body_span(mask: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size < 10:
        raise ValueError("Could not find a body silhouette")
    return int(rows[0]), int(rows[-1])


def detect_view(path: str, height_cm: float, view_name: str) -> PoseView:
    rgb = load_image(path)
    h, w = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result: PoseLandmarkerResult = get_landmarker().detect(mp_image)
    if not result.pose_landmarks:
        raise ValueError(f"Could not detect a person in the {view_name} photo")

    landmarks = result.pose_landmarks[0]
    needed = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"]
    missing = [name for name in needed if vis(landmarks, LM[name]) < 0.45]
    if len(missing) >= 3:
        raise ValueError(f"Body is not fully visible in the {view_name} photo")

    if not result.segmentation_masks:
        raise ValueError(f"Could not segment the body in the {view_name} photo")

    mask = result.segmentation_masks[0].numpy_view()
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    top, bottom = body_span(binary)
    px_height = max(bottom - top, 1)
    if px_height < h * 0.35:
        raise ValueError(f"Please stand farther back so the full body is in the {view_name} photo")

    margin = 8
    if top < margin or bottom > h - margin:
        # Head or feet clipped — scale is unreliable.
        raise ValueError(f"Head or feet are cut off in the {view_name} photo")

    scale = height_cm / float(px_height)
    return PoseView(
        image_rgb=rgb,
        mask=binary,
        landmarks=landmarks,
        width=w,
        height=h,
        scale_cm_per_px=scale,
        body_top=top,
        body_bottom=bottom,
    )


def midpoint(lm, a: str, b: str, width: int, height: int) -> tuple[float, float]:
    xa, ya = xy(lm, LM[a], width, height)
    xb, yb = xy(lm, LM[b], width, height)
    return (xa + xb) / 2.0, (ya + yb) / 2.0


def lerp_y(y0: float, y1: float, t: float) -> float:
    return y0 + (y1 - y0) * t


def accept(name: str, value: Optional[float], height_cm: float, conf: float) -> tuple[Optional[float], Optional[float]]:
    if value is None or not math.isfinite(value) or value <= 0:
        return None, None
    lo, hi = PLAUSIBLE[name]
    ratio = value / height_cm
    if ratio < lo or ratio > hi:
        return None, None
    if conf < 0.45:
        return None, None
    return round(value, 1), round(min(conf, 0.99), 2)


def measure(front: PoseView, side: PoseView, height_cm: float) -> dict:
    warnings: list[str] = []
    fl, sl = front.landmarks, side.landmarks
    fw, fh = front.width, front.height
    sw, sh = side.width, side.height

    estimates = {
        "height": round(height_cm, 1),
        "shoulderWidth": None,
        "chest": None,
        "waist": None,
        "hips": None,
        "upperArm": None,
        "thigh": None,
        "inseam": None,
        "unit": "cm",
        "confidence": {},
        "warnings": warnings,
    }

    # Shoulder width from front pose
    lsx, lsy = xy(fl, LM["left_shoulder"], fw, fh)
    rsx, rsy = xy(fl, LM["right_shoulder"], fw, fh)
    shoulder_px = math.hypot(lsx - rsx, lsy - rsy)
    shoulder_conf = min(vis(fl, LM["left_shoulder"]), vis(fl, LM["right_shoulder"]))
    shoulder_cm = shoulder_px * front.scale_cm_per_px
    estimates["shoulderWidth"], estimates["confidence"]["shoulderWidth"] = accept(
        "shoulderWidth", shoulder_cm, height_cm, 0.55 + 0.4 * shoulder_conf
    )
    if estimates["shoulderWidth"] is None:
        warnings.append("shoulderWidth")

    hip_y_f = midpoint(fl, "left_hip", "right_hip", fw, fh)[1]
    shoulder_y_f = (lsy + rsy) / 2.0
    hip_y_s = midpoint(sl, "left_hip", "right_hip", sw, sh)[1]
    shoulder_y_s = midpoint(sl, "left_shoulder", "right_shoulder", sw, sh)[1]

    def band(front_t: float, side_t: float) -> tuple[Optional[float], Optional[float], float]:
        fy = lerp_y(shoulder_y_f, hip_y_f, front_t)
        sy = lerp_y(shoulder_y_s, hip_y_s, side_t)
        front_w = mask_width_at(front.mask, int(fy))
        side_d = mask_width_at(side.mask, int(sy))
        if not front_w or not side_d:
            return None, None, 0.0
        width_cm = front_w * front.scale_cm_per_px
        depth_cm = side_d * side.scale_cm_per_px
        circ = ellipse_circumference(width_cm, depth_cm)
        conf = 0.82
        if abs(width_cm - depth_cm) / max(width_cm, depth_cm, 1) > 0.65:
            conf = 0.55
        return circ, max(width_cm, depth_cm), conf

    chest, _, chest_c = band(0.22, 0.22)
    estimates["chest"], estimates["confidence"]["chest"] = accept("chest", chest, height_cm, chest_c)
    if estimates["chest"] is None:
        warnings.append("chest")

    waist, _, waist_c = band(0.78, 0.78)
    estimates["waist"], estimates["confidence"]["waist"] = accept("waist", waist, height_cm, waist_c)
    if estimates["waist"] is None:
        warnings.append("waist")

    # Hips slightly below the hip landmarks
    hips_front_w = mask_width_at(front.mask, int(hip_y_f + 0.04 * fh))
    hips_side_d = mask_width_at(side.mask, int(hip_y_s + 0.04 * sh))
    hips = None
    hips_c = 0.0
    if hips_front_w and hips_side_d:
        hips = ellipse_circumference(
            hips_front_w * front.scale_cm_per_px,
            hips_side_d * side.scale_cm_per_px,
        )
        hips_c = 0.8
    estimates["hips"], estimates["confidence"]["hips"] = accept("hips", hips, height_cm, hips_c)
    if estimates["hips"] is None:
        warnings.append("hips")

    # Upper arm: mid shoulder-elbow, front width as diameter
    arm_ys = []
    for side_name in ("left", "right"):
        if vis(fl, LM[f"{side_name}_shoulder"]) >= 0.5 and vis(fl, LM[f"{side_name}_elbow"]) >= 0.5:
            _, sy0 = xy(fl, LM[f"{side_name}_shoulder"], fw, fh)
            _, ey0 = xy(fl, LM[f"{side_name}_elbow"], fw, fh)
            arm_ys.append((sy0 + ey0) / 2.0)
    upper = None
    upper_c = 0.0
    if arm_ys:
        widths = [mask_width_at(front.mask, int(y), search=4) for y in arm_ys]
        widths = [w for w in widths if w]
        if widths:
            # Arm is a fraction of torso width; take the smaller local band
            arm_px = min(widths)
            torso = mask_width_at(front.mask, int(shoulder_y_f + 0.15 * (hip_y_f - shoulder_y_f)))
            if torso and arm_px < torso * 0.45:
                upper = math.pi * arm_px * front.scale_cm_per_px
                upper_c = 0.62
    estimates["upperArm"], estimates["confidence"]["upperArm"] = accept("upperArm", upper, height_cm, upper_c)
    if estimates["upperArm"] is None:
        warnings.append("upperArm")

    # Thigh: 30% from hip to knee
    thigh = None
    thigh_c = 0.0
    knee_y = midpoint(fl, "left_knee", "right_knee", fw, fh)[1]
    thigh_y = lerp_y(hip_y_f, knee_y, 0.32)
    thigh_w = mask_width_at(front.mask, int(thigh_y), search=5)
    hip_w = mask_width_at(front.mask, int(hip_y_f))
    if thigh_w and hip_w and thigh_w < hip_w * 0.85:
        # One thigh is roughly half of the combined silhouette minus a gap;
        # if the mask still contains both legs, split.
        thigh_px = thigh_w / 2.0 if thigh_w > hip_w * 0.55 else thigh_w
        thigh_side = mask_width_at(side.mask, int(lerp_y(hip_y_s, midpoint(sl, "left_knee", "right_knee", sw, sh)[1], 0.32)))
        if thigh_side:
            thigh = ellipse_circumference(thigh_px * front.scale_cm_per_px, thigh_side * side.scale_cm_per_px)
            thigh_c = 0.68
        else:
            thigh = math.pi * thigh_px * front.scale_cm_per_px
            thigh_c = 0.5
    estimates["thigh"], estimates["confidence"]["thigh"] = accept("thigh", thigh, height_cm, thigh_c)
    if estimates["thigh"] is None:
        warnings.append("thigh")

    # Inseam: mid-hip to mid-ankle
    ankle_y = midpoint(fl, "left_ankle", "right_ankle", fw, fh)[1]
    inseam_px = abs(ankle_y - hip_y_f)
    inseam = inseam_px * front.scale_cm_per_px
    inseam_c = min(vis(fl, LM["left_ankle"]), vis(fl, LM["right_ankle"]), vis(fl, LM["left_hip"]), vis(fl, LM["right_hip"]))
    estimates["inseam"], estimates["confidence"]["inseam"] = accept("inseam", inseam, height_cm, 0.5 + 0.45 * inseam_c)
    if estimates["inseam"] is None:
        warnings.append("inseam")

    estimated_keys = [k for k in ("shoulderWidth", "chest", "waist", "hips", "upperArm", "thigh", "inseam") if estimates[k] is not None]
    if not estimated_keys:
        raise ValueError("Could not reliably estimate any body measurements from these photos")

    return estimates


def analyze_body(front_path: str, side_path: str, height_cm: float) -> dict:
    if height_cm < 100 or height_cm > 230:
        raise ValueError("Height must be between 100 and 230 cm")
    front = detect_view(front_path, height_cm, "front")
    side = detect_view(side_path, height_cm, "side")
    return measure(front, side, height_cm)
