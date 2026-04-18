import cv2
import numpy as np
import pickle
from sympy import sequence
import tensorflow as tf
import mediapipe as mp
from collections import deque
import os, sys, time, threading
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# ── Indic NLP ────────────────────────────────────────────────
from indicnlp import common
INDIC_NLP_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indic_nlp_resources')
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize

normalizer_factory = IndicNormalizerFactory()
hi_normalizer      = normalizer_factory.get_normalizer('hi')
mr_normalizer      = normalizer_factory.get_normalizer('mr')

from deep_translator import GoogleTranslator
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = 'NotoSansDevanagari.ttf'
try:
    deva_font = ImageFont.truetype(FONT_PATH, 20)
    print("✅ Devanagari font loaded")
except:
    deva_font = None
    print("⚠️  Devanagari font missing")

def put_devanagari(frame, text, pos, font, color=(255,255,255)):
    if not font or not text:
        return frame
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=tuple(color))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        return frame

def translate(text, lang_code):
    try:
        return GoogleTranslator(source='en', target=lang_code).translate(text)
    except Exception as e:
        return text

def normalize_indic(text, lang):
    try:
        n = hi_normalizer if lang == 'hi' else mr_normalizer
        return n.normalize(text)
    except:
        return text

def call_groq(words):
    """
    Single Groq API call — returns English + Hindi + Marathi
    with proper grammar in JSON format
    """
    import json

    prompt = f"""You are an Indian Sign Language (ISL) interpreter assistant.

These words were detected from ISL hand signs in sequence: {', '.join(words)}

Your tasks:
1. Form one natural, grammatically correct English sentence using these words
2. Translate it into Hindi with proper Devanagari script and grammar
3. Translate it into Marathi with proper Devanagari script and grammar

Important:
- Keep sentences simple and natural
- Use proper Hindi and Marathi grammar, not word-for-word translation
- Reply ONLY with JSON, no explanation, no markdown

JSON format:
{{
  "english": "formed English sentence",
  "hindi": "हिंदी वाक्य",
  "marathi": "मराठी वाक्य"
}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fast + free
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3                  # low temp = consistent output
        )
        raw_text = response.choices[0].message.content.strip()

        # Strip markdown if present
        raw_text = raw_text.replace('```json','').replace('```','').strip()

        result = json.loads(raw_text)
        return (
            result.get('english', ' '.join(words)),
            result.get('hindi',   ''),
            result.get('marathi', '')
        )

    except json.JSONDecodeError:
        print(f"⚠️  JSON parse error — raw: {raw_text}")
        return ' '.join(words), '', ''
    except Exception as e:
        print(f"⚠️  Groq error: {e}")
        return ' '.join(words), '', ''

# ── Load artifacts ───────────────────────────────────────────
print("\n[1/4] Loading model...")
try:
    model = tf.keras.models.load_model('best_model.keras')
    print(f"      ✅ input shape: {model.input_shape}")
except Exception as e:
    print(f"      ❌ {e}"); sys.exit()

print("[2/4] Loading label encoder...")
try:
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print(f"      ✅ {len(le.classes_)} classes")
except Exception as e:
    print(f"      ❌ {e}"); sys.exit()

print("[3/4] Loading norm stats...")
try:
    mean = np.load('norm_mean.npy')
    std  = np.load('norm_std.npy')
    feat = mean.shape[-1]
    print(f"      ✅ features: {feat}")
except Exception as e:
    print(f"      ❌ {e}"); sys.exit()

print("[4/4] Loading MediaPipe...")
try:
    BaseOptions         = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticOptions    = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode
    options = HolisticOptions(
        base_options=BaseOptions(model_asset_path='holistic_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        min_face_detection_confidence=0.3,
        min_pose_detection_confidence=0.3,
        min_hand_landmarks_confidence=0.3
    )
    landmarker = HolisticLandmarker.create_from_options(options)
    print(f"      ✅ ready")
except Exception as e:
    print(f"      ❌ {e}"); sys.exit()

# ── Landmark drawing ─────────────────────────────────────────
POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),
             (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
             (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

def draw_landmarks(frame, result, h, w):
    if result.pose_landmarks:
        pts = {i: (int(lm.x*w), int(lm.y*h))
               for i, lm in enumerate(result.pose_landmarks)}
        for a,b in POSE_CONN:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0,200,200), 2)
        for p in pts.values():
            cv2.circle(frame, p, 4, (0,255,255), -1)

    if result.left_hand_landmarks:
        pts = {i: (int(lm.x*w), int(lm.y*h))
               for i, lm in enumerate(result.left_hand_landmarks)}
        for a,b in HAND_CONN:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (200,80,0), 2)
        for p in pts.values():
            cv2.circle(frame, p, 5, (255,100,0), -1)

    if result.right_hand_landmarks:
        pts = {i: (int(lm.x*w), int(lm.y*h))
               for i, lm in enumerate(result.right_hand_landmarks)}
        for a,b in HAND_CONN:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0,80,200), 2)
        for p in pts.values():
            cv2.circle(frame, p, 5, (0,100,255), -1)

def extract_keypoints(result):
    pose = np.array([[lm.x,lm.y,lm.z] for lm in result.pose_landmarks]).flatten() \
           if result.pose_landmarks else np.zeros(99)
    lh   = np.array([[lm.x,lm.y,lm.z] for lm in result.left_hand_landmarks]).flatten() \
           if result.left_hand_landmarks else np.zeros(63)
    rh   = np.array([[lm.x,lm.y,lm.z] for lm in result.right_hand_landmarks]).flatten() \
           if result.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# ── State ─────────────────────────────────────────────────────
MAX_FRAMES        = 30
PREDICT_EVERY     = 8
CONFIDENCE_THRESH = 0.30        # ← lowered from 0.45
WORD_COOLDOWN     = 20

BUFFER          = deque(maxlen=MAX_FRAMES)
detected_words  = []
sentence_en     = ""
sentence_hi     = ""
sentence_mr     = ""
prediction      = "Warming up..."
confidence      = 0.0
top3            = []
frame_count     = 0
last_word_frame = 0
last_word       = ""
fps_time        = time.time()
fps             = 0
nlp_processing  = False

def run_nlp(words):
    global sentence_en, sentence_hi, sentence_mr, nlp_processing
    nlp_processing = True
    print(f"\n🧠 Sending to Groq: {words}")

    en, hi, mr = call_groq(words)

    sentence_en = en
    sentence_hi = normalize_indic(hi, 'hi') if hi else ''
    sentence_mr = normalize_indic(mr, 'mr') if mr else ''

    print(f"   EN : {sentence_en}")
    print(f"   HI : {sentence_hi}")
    print(f"   MR : {sentence_mr}")

    nlp_processing = False

# ── Webcam ───────────────────────────────────────────────────
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found!"); sys.exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

PANEL_W = 450     # right side panel width
print("✅ Ready — SPACE=Process | C=Clear | Q=Quit\n")
def wrap_text_pil(text, font, max_width):
    words = text.split()
    lines = []
    line = ""

    for word in words:
        test_line = (line + " " + word).strip()
        w, h = font.getbbox(test_line)[2:4]

        if w <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    return lines

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    frame_count += 1

    # ── MediaPipe ────────────────────────────────────────────
    try:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)
        draw_landmarks(frame, result, h, w)
        kp = extract_keypoints(result)
        BUFFER.append(kp)
    except:
        BUFFER.append(np.zeros(feat))

    # ── Prediction ───────────────────────────────────────────
    if frame_count % PREDICT_EVERY == 0 and len(BUFFER) == MAX_FRAMES:
        try:
            seq   = np.array(BUFFER)
            seq   = (seq - mean.squeeze()) / std.squeeze()
            seq   = np.expand_dims(seq, axis=0)
            probs = model.predict(seq, verbose=0)[0]

            top3_idx   = np.argsort(probs)[::-1][:3]
            prediction = le.classes_[top3_idx[0]]
            confidence = float(probs[top3_idx[0]])
            top3       = [(le.classes_[i], float(probs[i])) for i in top3_idx]

            if (confidence >= CONFIDENCE_THRESH and
                    prediction != last_word and
                    frame_count - last_word_frame > WORD_COOLDOWN and
                    len(detected_words) < 8):
                detected_words.append(prediction)
                last_word       = prediction
                last_word_frame = frame_count
                print(f"✅ Word: '{prediction}' ({confidence*100:.0f}%)")
        except Exception as e:
            print(f"Pred error: {e}")

    # ── FPS ──────────────────────────────────────────────────
    if frame_count % 30 == 0:
        fps      = 30 / (time.time() - fps_time)
        fps_time = time.time()

    # ════════════════════════════════════════════════════════
    # LAYOUT: camera frame LEFT | info panel RIGHT
    # ════════════════════════════════════════════════════════
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    # ── Prediction on camera frame (top left, small) ─────────
    color = (0,255,80) if confidence>0.6 else \
            (0,200,255) if confidence>0.35 else (120,120,255)
    cv2.rectangle(frame, (0,0), (w, 40), (0,0,0), -1)
    cv2.putText(frame, f"{prediction.upper()}  {confidence*100:.0f}%",
                (8, 28), cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)

    # Buffer bar on camera frame (bottom, thin)
    filled = int((len(BUFFER)/MAX_FRAMES)*w)
    cv2.rectangle(frame, (0, h-5), (w, h), (40,40,40), -1)
    cv2.rectangle(frame, (0, h-5), (filled, h), (0,180,0), -1)

    # FPS on camera frame
    cv2.putText(frame, f"FPS:{fps:.0f}", (w-65, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

    # ── RIGHT PANEL content ───────────────────────────────────
    py = 15

    # Title
    cv2.putText(panel, "ISL DETECTOR", (10, py+15),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (100,200,255), 1)
    py += 40
    cv2.line(panel, (5,py), (PANEL_W-5,py), (60,60,60), 1)
    py += 15

    # Top 3 predictions
    cv2.putText(panel, "TOP 3 PREDICTIONS", (10,py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
    py += 18
    for i, (word, prob) in enumerate(top3 if top3 else [('—',0)]*3):
        bl    = int((PANEL_W-20) * prob)
        bcol  = (0,180,70) if i==0 else (60,60,150)
        cv2.rectangle(panel, (10,py), (10+bl, py+18), bcol, -1)
        cv2.putText(panel, f"{word}  {prob*100:.0f}%",
                    (12, py+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
        py += 24
    py += 5
    cv2.line(panel, (5,py), (PANEL_W-5,py), (60,60,60), 1)
    py += 15

    # Detected words
    cv2.putText(panel, "DETECTED WORDS", (10,py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
    py += 20

    # Word bubbles
    x_off = 10
    for word in detected_words:
        tw     = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        box_w  = tw + 10
        if x_off + box_w > PANEL_W - 10:
            x_off = 10
            py   += 24
        cv2.rectangle(panel, (x_off, py-14),
                      (x_off+box_w, py+4), (60,80,120), -1)
        cv2.putText(panel, word, (x_off+5, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
        x_off += box_w + 6

    py += 30
    cv2.line(panel, (5,py), (PANEL_W-5,py), (60,60,60), 1)
    py += 15

    # English sentence
    cv2.putText(panel, "ENGLISH", (10,py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,200,255), 1)
    py += 18

    # Word wrap English
    en_text = sentence_en if sentence_en else "Press SPACE to generate"
    words_en = en_text.split()
    line, lines_en = "", []
    for wd in words_en:
        test = (line + " " + wd).strip()
        if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] < PANEL_W-20:
            line = test
        else:
            lines_en.append(line)
            line = wd
    lines_en.append(line)
    for ln in lines_en:
        cv2.putText(panel, ln, (10,py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)
        py += 18
    py += 5
    cv2.line(panel, (5,py), (PANEL_W-5,py), (60,60,60), 1)
    py += 15

    # Hindi
    cv2.putText(panel, "HINDI", (10,py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,255,150), 1)
    py += 18
    
    if sentence_hi:
        lines_hi = wrap_text_pil(sentence_hi, deva_font, PANEL_W - 20)

        for ln in lines_hi:
            panel = put_devanagari(panel, ln, (10, py), deva_font, (220,255,220))
            py += 22
    else:
        cv2.putText(panel, "—", (10,py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1)
        py += 20

    # Marathi
    cv2.putText(panel, "MARATHI", (10,py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,180,100), 1)
    py += 18
    if sentence_mr:
        lines_mr = wrap_text_pil(sentence_mr, deva_font, PANEL_W - 20)

        for ln in lines_mr:
            panel = put_devanagari(panel, ln, (10, py), deva_font, (255,220,180))
            py += 22
    else:
        cv2.putText(panel, "—", (10,py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1)
        py += 20

    # Controls at bottom of panel
    ctrl_y = h - 70
    cv2.putText(panel, "CONTROLS", (10, ctrl_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
    cv2.putText(panel, "SPACE  →  Process sentence",
                (10, ctrl_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
    cv2.putText(panel, "C      →  Clear all",
                (10, ctrl_y+34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
    cv2.putText(panel, "Q      →  Quit",
                (10, ctrl_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)

    if nlp_processing:
        cv2.putText(panel, "Processing...",
                    (10, ctrl_y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

    # ── Combine camera + panel side by side ──────────────────
    combined = np.hstack([frame, panel])
    cv2.imshow('ISL Detector + NLP + Indic', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        detected_words.clear()
        sentence_en = sentence_hi = sentence_mr = ""
        last_word = ""
        BUFFER.clear()
        print("🔄 Cleared")
    elif key == ord(' '):
        if detected_words and not nlp_processing:
            t = threading.Thread(target=run_nlp, args=(detected_words.copy(),))
            t.daemon = True
            t.start()
        elif not detected_words:
            print("⚠️  No words detected yet")

landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("\n👋 Done.")