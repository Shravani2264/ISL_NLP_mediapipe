import cv2
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
from collections import deque
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 50)
print("   ISL — Indian Sign Language Detector")
print("        (MediaPipe Holistic Edition)")
print("=" * 50)

# ── Load artifacts ───────────────────────────────────────────
print("\n[1/4] Loading model...")
try:
    model = tf.keras.models.load_model('best_model.keras')
    print(f"      ✅ Model loaded — input: {model.input_shape}")
except Exception as e:
    print(f"      ❌ {e}"); sys.exit()

print("[2/4] Loading label encoder...")
try:
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print(f"      ✅ {len(le.classes_)} classes")
except Exception as e:
    print(f"      ❌ {e}"); sys.exit()

print("[3/4] Loading normalization stats...")
try:
    mean = np.load('norm_mean.npy')
    std  = np.load('norm_std.npy')
    print(f"      ✅ mean shape: {mean.shape}")
    if mean.shape[-1] != 225:
        print(f"      ❌ Wrong norm files! Expected (1,1,225), got {mean.shape}")
        print(f"         Download norm files from the MediaPipe Colab session.")
        sys.exit()
except Exception as e:
    print(f"      ❌ {e}"); sys.exit()

print("[4/4] Loading MediaPipe Holistic...")
try:
    BaseOptions         = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticOptions    = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode

    options = HolisticOptions(
        base_options=BaseOptions(model_asset_path='holistic_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )
    landmarker = HolisticLandmarker.create_from_options(options)
    print(f"      ✅ MediaPipe Holistic ready")
except Exception as e:
    print(f"      ❌ MediaPipe failed: {e}")
    sys.exit()

# ── Drawing connections ──────────────────────────────────────
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28)
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, result, h, w):
    # Pose
    if result.pose_landmarks:
        pts = {}
        for i, lm in enumerate(result.pose_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            pts[i] = (x, y)
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)

    # Left hand
    if result.left_hand_landmarks:
        pts = {}
        for i, lm in enumerate(result.left_hand_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            pts[i] = (x, y)
            cv2.circle(frame, (x, y), 5, (255, 100, 0), -1)
        for a, b in HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (200, 80, 0), 2)

    # Right hand
    if result.right_hand_landmarks:
        pts = {}
        for i, lm in enumerate(result.right_hand_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            pts[i] = (x, y)
            cv2.circle(frame, (x, y), 5, (0, 100, 255), -1)
        for a, b in HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 80, 200), 2)

# ── MediaPipe feature extraction (matches training) ──────────
def extract_keypoints(result):
    # Pose: 33 x 3 = 99
    if result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in result.pose_landmarks]).flatten()
    else:
        pose = np.zeros(33 * 3)

    # Left hand: 21 x 3 = 63
    if result.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in result.left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right hand: 21 x 3 = 63
    if result.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in result.right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])  # 225 features

# ── Config ───────────────────────────────────────────────────
MAX_FRAMES    = 30
PREDICT_EVERY = 10
BUFFER        = deque(maxlen=MAX_FRAMES)

# ── Webcam ───────────────────────────────────────────────────
print("\n[Webcam] Opening...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found! Try VideoCapture(1)")
    sys.exit()
print("[Webcam] ✅ Ready\n")
print("🟢 ISL Detector running")
print("   Q = Quit | C = Clear buffer\n")

prediction  = "Warming up..."
confidence  = 0.0
top3        = []
frame_count = 0
fps_time    = time.time()
fps         = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    frame_count += 1

    # ── MediaPipe detection (features + visuals) ─────────────
    try:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        # Draw body tracking
        draw_landmarks(frame, result, h, w)

        # Extract features for prediction
        keypoints = extract_keypoints(result)
        BUFFER.append(keypoints)

    except Exception as e:
        BUFFER.append(np.zeros(225))

    # ── Predict every N frames ───────────────────────────────
    if frame_count % PREDICT_EVERY == 0 and len(BUFFER) == MAX_FRAMES:
        try:
            sequence = np.array(BUFFER)               # (30, 225)
            sequence = (sequence - mean.squeeze()) / std.squeeze()   # (30, 225)        # normalize
            sequence = np.expand_dims(sequence, axis=0)              # (1, 30, 225)
            probs    = model.predict(sequence, verbose=0)[0]

            top3_idx   = np.argsort(probs)[::-1][:3]
            prediction = le.classes_[top3_idx[0]]
            confidence = float(probs[top3_idx[0]])
            top3       = [(le.classes_[i], float(probs[i])) for i in top3_idx]
        except Exception as e:
            print(f"Prediction error: {e}")

    # ── FPS ──────────────────────────────────────────────────
    if frame_count % 30 == 0:
        fps      = 30 / (time.time() - fps_time)
        fps_time = time.time()

    # ── UI Overlay ───────────────────────────────────────────
    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Prediction color based on confidence
    color = (0, 255, 80)   if confidence > 0.6  else \
            (0, 200, 255)  if confidence > 0.35 else \
            (100, 100, 255)

    # Main prediction
    cv2.putText(frame, prediction.upper(),
                (12, 50), cv2.FONT_HERSHEY_DUPLEX,
                1.5, color, 2)

    # Confidence %
    if confidence > 0:
        cv2.putText(frame, f"{confidence*100:.0f}%",
                    (w - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 0), 2)

    # Top 3 panel (bottom left)
    if top3:
        cv2.rectangle(frame, (0, h-100), (230, h), (20,20,20), -1)
        cv2.putText(frame, "TOP PREDICTIONS",
                    (8, h-80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180,180,180), 1)
        for i, (word, prob) in enumerate(top3):
            bar_len   = int(200 * prob)
            bar_color = (0,200,80) if i == 0 else (100,100,200)
            cv2.rectangle(frame,
                          (8, h-62+i*22),
                          (8+bar_len, h-50+i*22),
                          bar_color, -1)
            cv2.putText(frame, f"{word}  {prob*100:.0f}%",
                        (10, h-52+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255,255,255), 1)

    # Buffer fill bar
    filled = int((len(BUFFER) / MAX_FRAMES) * w)
    cv2.rectangle(frame, (0, h-6), (w, h), (40,40,40), -1)
    cv2.rectangle(frame, (0, h-6), (filled, h), (0,180,0), -1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w-80, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (150,150,150), 1)

    # Landmark legend
    cv2.circle(frame, (w-200, h-55), 5, (0,255,255), -1)
    cv2.putText(frame, "Pose",       (w-190, h-51), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    cv2.circle(frame, (w-200, h-38), 5, (255,100,0), -1)
    cv2.putText(frame, "Left Hand",  (w-190, h-34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    cv2.circle(frame, (w-200, h-21), 5, (0,100,255), -1)
    cv2.putText(frame, "Right Hand", (w-190, h-17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

    cv2.imshow('ISL Detector — Indian Sign Language', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        BUFFER.clear()
        prediction = "Cleared!"
        print("🔄 Buffer cleared")

landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("\n👋 ISL Detector closed.")