"""
Live Data Collector - Collects ASL landmarks from your webcam via browser
Adds to existing landmarks_fixed.csv so you keep the kaggle data too.

Run: python collect_data.py
Then open http://127.0.0.1:8001 in browser
"""

import os
import csv
import base64
import numpy as np
import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── CONFIG ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(BASE_DIR, "landmarks", "landmarks_live.csv")
TASK_PATH   = os.path.join(BASE_DIR, "hand_landmarker.task")
LABELS      = list("abcdefghijklmnopqrstuvwxyz") + list("0123456789")

# ── MEDIAPIPE ────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=TASK_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
detector = vision.HandLandmarker.create_from_options(options)

# ── CSV SETUP ────────────────────────────────────────────
cols = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    print(f"Created {OUTPUT_CSV}")

collected = {l: 0 for l in LABELS}

# ── FASTAPI ──────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(img)
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    xs = [pt.x for pt in hand]
    ys = [pt.y for pt in hand]
    return xs + ys


def decode_image(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.post("/collect")
async def collect(request: Request):
    data = await request.json()
    label = data.get("label", "").strip().lower()
    image = data.get("image")

    if label not in LABELS:
        return {"ok": False, "error": f"Invalid label '{label}'"}

    frame = decode_image(image)
    if frame is None:
        return {"ok": False, "error": "Bad image"}

    landmarks = extract_landmarks(frame)
    if landmarks is None:
        return {"ok": False, "error": "No hand detected"}

    row = [label] + landmarks
    with open(OUTPUT_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)

    collected[label] += 1
    print(f"  [{label}] {collected[label]} samples")
    return {"ok": True, "label": label, "count": collected[label]}


@app.get("/counts")
def counts():
    return collected


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html>
<head>
<title>ASL Data Collector</title>
<style>
  body { font-family: sans-serif; background: #111; color: #eee; display: flex; flex-direction: column; align-items: center; padding: 20px; }
  video { border: 2px solid #444; border-radius: 8px; width: 480px; }
  .controls { margin: 16px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center; }
  select, button { padding: 10px 20px; font-size: 16px; border-radius: 6px; border: none; cursor: pointer; }
  button { background: #4C9BE8; color: white; }
  button:active { background: #2a7bc4; }
  #status { font-size: 18px; margin: 10px; min-height: 30px; }
  #counts { display: flex; flex-wrap: wrap; gap: 6px; max-width: 600px; justify-content: center; margin-top: 10px; }
  .badge { background: #222; border: 1px solid #444; padding: 4px 10px; border-radius: 20px; font-size: 13px; }
  .badge.has-data { border-color: #4C9BE8; color: #4C9BE8; }
</style>
</head>
<body>
<h2>ASL Live Data Collector</h2>
<video id="vid" autoplay playsinline></video>
<div class="controls">
  <select id="label">
    <option value="">-- Pick a letter --</option>
    <option>a</option><option>b</option><option>c</option><option>d</option>
    <option>e</option><option>f</option><option>g</option><option>h</option>
    <option>i</option><option>j</option><option>k</option><option>l</option>
    <option>m</option><option>n</option><option>o</option><option>p</option>
    <option>q</option><option>r</option><option>s</option><option>t</option>
    <option>u</option><option>v</option><option>w</option><option>x</option>
    <option>y</option><option>z</option>
    <option>0</option><option>1</option><option>2</option><option>3</option>
    <option>4</option><option>5</option><option>6</option><option>7</option>
    <option>8</option><option>9</option>
  </select>
  <button onclick="capture()">Capture (Space)</button>
  <button onclick="startAuto()" id="autobtn">Auto (hold sign)</button>
</div>
<div id="status">Pick a letter and show your hand</div>
<div id="counts"></div>

<script>
const vid = document.getElementById('vid');
const labelSel = document.getElementById('label');
const status = document.getElementById('status');
let autoInterval = null;

navigator.mediaDevices.getUserMedia({ video: true }).then(s => vid.srcObject = s);

function getFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = vid.videoWidth;
  canvas.height = vid.videoHeight;
  canvas.getContext('2d').drawImage(vid, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.8);
}

async function capture() {
  const label = labelSel.value;
  if (!label) { status.textContent = 'Pick a letter first!'; return; }
  const res = await fetch('/collect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, image: getFrame() })
  });
  const data = await res.json();
  if (data.ok) {
    status.textContent = `[${label.toUpperCase()}] ${data.count} samples saved`;
    updateCounts();
  } else {
    status.textContent = 'Error: ' + data.error;
  }
}

function startAuto() {
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
    document.getElementById('autobtn').textContent = 'Auto (hold sign)';
    return;
  }
  document.getElementById('autobtn').textContent = 'Stop Auto';
  autoInterval = setInterval(capture, 300);
}

async function updateCounts() {
  const res = await fetch('/counts');
  const data = await res.json();
  const div = document.getElementById('counts');
  div.innerHTML = Object.entries(data).map(([l, c]) =>
    `<span class="badge ${c > 0 ? 'has-data' : ''}">${l.toUpperCase()}: ${c}</span>`
  ).join('');
}

document.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); capture(); } });
updateCounts();
setInterval(updateCounts, 3000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("=" * 50)
    print("  ASL Data Collector")
    print("  Open http://127.0.0.1:8001 in browser")
    print("  Collect 100+ samples per letter")
    print("  Then retrain with train_models.py")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001)
