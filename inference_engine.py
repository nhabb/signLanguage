import pickle
import numpy as np
import cv2
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#here i have used pickle as a serialization library to import the models already saved as binary format
with open(os.path.join(BASE_DIR, "output/models/random_forest.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "output/models/scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, "output/models/label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

print(f"Model loaded | features expected: {model.n_features_in_}")

# i imported the mediapipe model so i can create a detector object 
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    # load the model 
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    # limit the number of hand detection to 1 for improved stability
    num_hands=1,
    # minimum confidence required to accept and keep track of the hand 
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
#create the detector
detector = vision.HandLandmarker.create_from_options(options)

#used to make predicitions independant from hand position and hand size inside the frame
def normalize_landmarks(xs, ys):
    #convert coordinates into numpy array for better math operations 
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # use xs[0] ys[0] the start of the wrist as the center of origin
    #that way we can ensure a dynamic position of the hand 
    cx, cy = xs[0], ys[0]
    xs -= cx
    ys -= cy

    # Scale by wrist to middle finger MCP (point 9) this will help us scale the input
    #so any hand close or far from the camera will be treated similarly 
    scale = np.sqrt(xs[9]**2 + ys[9]**2)
    if scale > 0:
        xs /= scale
        ys /= scale

    #create the vector that will be directly inputed in the random forest model
    # the result vector will have a format of [x0,y0,x1,y1,x2,y2....] which is identical to the dataset used in training 
    result = []
    for i in range(21): #range 21 because we have 21 features of x and y 
        result.append(float(xs[i]))
        result.append(float(ys[i]))
    return result


def extract_landmarks(frame):
    #since mediapipe expects RGB input we needed to convert the frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #create a media pipe image for the detector 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    #run the detector on the image 
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    #we took [0] here because in the detector construction we specified the use of 1 hand only 
    hand = result.hand_landmarks[0]
    #we here extract all the x and y coordinates from the frame 
    xs = [pt.x for pt in hand]
    ys = [pt.y for pt in hand]
    #finally we normalize them and return the feature vector used in the call of the random forest model
    return normalize_landmarks(xs, ys)

#this function is the main prediction function that i used to call the random forest model and predict the sign 
def predict_from_frame(frame):
    #use the previous functions to extract and normalize landmarks and return the feature vector 
    landmarks = extract_landmarks(frame)
    if landmarks is None:
        return None, 0.0
    
    # x is now an numpy array containing the 42 features
    x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
    
    x = scaler.transform(x)
   

    #predict the class using the random forest model 
    pred = model.predict(x)[0]

    #predict the probability of each class
    proba = model.predict_proba(x)[0]

    #convert the prediction into the lable 
    label = label_encoder.inverse_transform([pred])[0]
    return label, float(np.max(proba))