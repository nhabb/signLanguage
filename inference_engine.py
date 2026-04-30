import pickle
import numpy as np
import cv2
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "output/models/random_forest.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "output/models/scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, "output/models/label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

print(f"Model loaded | features expected: {model.n_features_in_}")

MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
detector = vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(xs, ys):
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # Center on wrist
    cx, cy = xs[0], ys[0]
    xs -= cx
    ys -= cy

    # Scale by wrist to middle finger MCP (point 9)
    scale = np.sqrt(xs[9]**2 + ys[9]**2)
    if scale > 0:
        xs /= scale
        ys /= scale

    # Interleave: x0,y0,x1,y1,...
    result = []
    for i in range(21):
        result.append(float(xs[i]))
        result.append(float(ys[i]))
    return result


def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]
    xs = [pt.x for pt in hand]
    ys = [pt.y for pt in hand]

    return normalize_landmarks(xs, ys)


def predict_from_frame(frame):
    landmarks = extract_landmarks(frame)
    if landmarks is None:
        return None, 0.0

    x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
    x = scaler.transform(x)
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label, float(np.max(proba))