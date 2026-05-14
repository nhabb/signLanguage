from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64

from inference_engine import predict_from_frame

app = FastAPI()

#to run the server:   uvicorn server:app --reload

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(base64_str):
    try:
        #return if no data received 
        if base64_str is None:
            return None

        # remove metadata header if exists
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        #convert the image into raw bytes
        img_bytes = base64.b64decode(base64_str)
        #create a numpy array that will be used in the prediction
        np_arr = np.frombuffer(img_bytes, np.uint8)

        if len(np_arr) == 0:
            return None

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return frame

    except Exception as e:
        print("Decode error:", e)
        return None

@app.get("/")
def home():
    return {"status": "ASL server running"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    frame = decode_image(data.get("image"))

    if frame is None:
        return {
            "letter": None,
            "confidence": 0.0,
            "error": "empty frame"
        }

    pred, conf = predict_from_frame(frame)

    return {
        "letter": pred,
        "confidence": conf
    }