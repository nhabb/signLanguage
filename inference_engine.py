import pickle
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================================
# LOAD MODEL
# ================================
with open("output/models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("output/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("output/models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Model loaded")


# ================================
# MEDIAPIPE SETUP
# ================================
MODEL_PATH = "hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)

detector = vision.HandLandmarker.create_from_options(options)


# ================================
# EXTRACT LANDMARKS (MATCH TRAINING)
# ================================
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    landmarks = []

    for pt in hand:
        landmarks.append(pt.x)
        landmarks.append(pt.y)

    return landmarks

# ================================
# PREDICTION
# ================================
def predict_from_landmarks(landmarks):
    x = np.array(landmarks).reshape(1, -1)

    # ONLY scaler (NO normalization!)
    x = scaler.transform(x)

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]

    label = label_encoder.inverse_transform([pred])[0]

    return label, float(np.max(proba))


# ================================
# MAIN FUNCTION (USED BY SERVER)
# ================================
def predict_from_frame(frame):
    landmarks = extract_landmarks(frame)

    if landmarks is None:
        return None, 0.0


    return predict_from_landmarks(landmarks)