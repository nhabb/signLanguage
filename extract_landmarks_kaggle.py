"""
Hand Landmark Extraction for Sign Language Dataset
Run on Kaggle — outputs /kaggle/working/landmarks.csv

Change DATASET_DIR to match your dataset path.
"""

import subprocess
subprocess.run(["pip", "install", "mediapipe", "-q"], check=True)
subprocess.run(["wget", "-q", "-O", "/kaggle/working/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"],
    check=True)

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = "/kaggle/input/datasets/vignonantoine/mediapipe-processed-asl-dataset/processed_combine_asl_dataset"
OUTPUT_CSV  = "/kaggle/working/landmarks.csv"
MODEL_PATH  = "/kaggle/working/hand_landmarker.task"
MODE        = "mediapipe"   # "mediapipe" | "color"
#   mediapipe = real hand photos
#   color     = pre-rendered colored-dot images (like the ones in your dataset)
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────

_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
DETECTOR = vision.HandLandmarker.create_from_options(_options)

LM_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def extract_mediapipe(img_bgr):
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = DETECTOR.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    return [v for pt in lm for v in (pt.x, pt.y, pt.z)]


# Color ranges (HSV) for pre-rendered landmark dot images
COLOR_GROUPS = [
    (np.array([0,   50, 150]), np.array([10,  255, 255]), [1, 2, 3, 4]),    # red   → thumb
    (np.array([10,  50, 150]), np.array([25,  255, 255]), [0]),              # orange → wrist
    (np.array([25,  50, 150]), np.array([35,  255, 255]), [5, 6, 7, 8]),    # yellow → index
    (np.array([35,  50, 100]), np.array([85,  255, 255]), [9, 10, 11, 12]), # green  → middle
    (np.array([85,  50, 100]), np.array([130, 255, 255]), [13, 14, 15, 16]),# blue   → ring
    (np.array([130, 50, 100]), np.array([160, 255, 255]), [17, 18, 19, 20]),# purple → pinky
]

def extract_color(img_bgr):
    hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w   = img_bgr.shape[:2]
    coords = {}
    for lo, hi, indices in COLOR_GROUPS:
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = sorted(
            [cv2.moments(c) for c in cnts if cv2.contourArea(c) > 5],
            key=lambda m: m["m10"] / m["m00"] if m["m00"] else 0,
        )
        for idx, mom in zip(indices, centers):
            if mom["m00"] == 0:
                continue
            coords[idx] = (mom["m10"] / mom["m00"] / w,
                           mom["m01"] / mom["m00"] / h,
                           0.0)
    if len(coords) < 10:
        return None
    row = []
    for i in range(21):
        row.extend(coords.get(i, (0.0, 0.0, 0.0)))
    return row


def process_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    if MODE == "color":
        return extract_color(img)
    result = extract_mediapipe(img)
    return result if result is not None else extract_color(img)


def main():
    records, skipped = [], 0
    dataset   = Path(DATASET_DIR)
    label_dirs = sorted([d for d in dataset.iterdir() if d.is_dir()]) or [dataset]

    for label_dir in label_dirs:
        images = [f for f in label_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        print(f"[{label_dir.name}] {len(images)} images")
        for path in images:
            lm = process_image(path)
            if lm is None:
                skipped += 1
                continue
            records.append({"label": label_dir.name, "file": path.name,
                            **dict(zip(LM_COLS, lm))})

    df = pd.DataFrame(records, columns=["label", "file"] + LM_COLS)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}  (skipped {skipped})")
    print(df.head())

main()
