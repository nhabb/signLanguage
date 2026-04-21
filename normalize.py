import numpy as np

def normalize_landmarks(hand_landmarks):
    # Step 1: extract all 21 points into a numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    # Step 2: translate — make wrist (point 0) the origin
    wrist = points[0]
    points = points - wrist

    # Step 3: scale — find the farthest point from the wrist
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances)

    # avoid division by zero just in case
    if max_distance > 0:
        points = points / max_distance

    # Step 4: flatten into one row of 63 numbers
    return points.flatten()

# quick test with fake data
import mediapipe as mp

class FakeLM:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

fake_hand = [FakeLM(i*0.05, i*0.03, i*0.01) for i in range(21)]
result = normalize_landmarks(fake_hand)

print(f"Output length: {len(result)}")   # should be 63
print(f"First 6 values: {result[:6]}")  # first two points
print(f"Max value: {result.max():.4f}") # should be close to 1.0