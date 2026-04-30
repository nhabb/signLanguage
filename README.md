# ASL Sign Language Recognition

## Setup

Install dependencies (Python 3.12 required):
```
C:\Users\rajha\AppData\Local\Python\bin\python.exe -m pip install mediapipe opencv-python fastapi uvicorn numpy scikit-learn pandas matplotlib seaborn
```

## Train the model (2-3 minutes)
```
C:\Users\rajha\AppData\Local\Python\bin\python.exe train_models.py
```

## Run the server
```
C:\Users\rajha\AppData\Local\Python\bin\python.exe -m uvicorn server:app --reload
```

Then open `index.html` in your browser.

## Project Structure
```
/
├── train_models.py        # Train RF + MLP models
├── inference_engine.py    # Mediapipe + model prediction
├── server.py              # FastAPI server
├── index.html             # Frontend
├── hand_landmarker.task   # Mediapipe model
├── landmarks/
│   └── landmarks_fixed.csv
└── output/
    └── models/            # Generated after training
```
