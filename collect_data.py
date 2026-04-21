import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from normalize import normalize_landmarks

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

SAMPLES_PER_SIGN = 150
OUTPUT_FILE = "landmarks.csv"

# build column names: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

# load existing data if file already exists
if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE)
    all_rows = df_existing.values.tolist()
    print(f"Loaded {len(all_rows)} existing samples")
else:
    all_rows = []

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        sign = input("\nEnter the sign to record (or 'quit' to finish): ").strip().upper()
        if sign == "QUIT":
            break

        # countdown
        print(f"Get ready to sign '{sign}'...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("  GO! Recording...")

        collected = 0
        while collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            # show progress on the webcam window
            cv2.putText(frame, f"Sign: {sign}  Samples: {collected}/{SAMPLES_PER_SIGN}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collecting", frame)
            cv2.waitKey(1)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                normalized = normalize_landmarks(hand)
                row = list(normalized) + [sign]
                all_rows.append(row)
                collected += 1

        print(f"  Done! Collected {collected} samples for '{sign}'")

# save everything to CSV
df = pd.DataFrame(all_rows, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df)} total rows to {OUTPUT_FILE}")

cap.release()
cv2.destroyAllWindows()