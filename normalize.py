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

