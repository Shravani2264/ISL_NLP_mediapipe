import cv2
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
import os, sys, time, threading, json
from groq import Groq
from dotenv import load_dotenv
from PIL import ImageFont, ImageDraw, Image
from indicnlp import common
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Groq ─────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# ── Indic NLP ────────────────────────────────────────────────
INDIC_NLP_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indic_nlp_resources')
common.set_resources_path(INDIC_NLP_RESOURCES)
normalizer_factory = IndicNormalizerFactory()
hi_normalizer      = normalizer_factory.get_normalizer('hi')
mr_normalizer      = normalizer_factory.get_normalizer('mr')

def normalize_indic(text, lang):
    try:
        n = hi_normalizer if lang == 'hi' else mr_normalizer
        return n.normalize(text)
    except:
        return text

# ── Groq call ─────────────────────────────────────────────────
def call_groq(words):
    prompt = f"""You are an Indian Sign Language (ISL) interpreter assistant.

These words were detected from ISL hand signs in sequence: {', '.join(words)}

Your tasks:
1. Form one natural, grammatically correct English sentence using these words
2. Translate it into Hindi with proper Devanagari script and natural grammar
3. Translate it into Marathi with proper Devanagari script and natural grammar

Rules:
- Keep sentences short and natural
- Use proper Hindi and Marathi grammar, not word-for-word translation
- Reply ONLY with JSON, absolutely no explanation or markdown

JSON format:
{{
  "english": "formed English sentence",
  "hindi": "हिंदी वाक्य",
  "marathi": "मराठी वाक्य"
}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3
        )
        raw  = response.choices[0].message.content.strip()
        raw  = raw.replace('```json','').replace('```','').strip()
        res  = json.loads(raw)
        return (
            res.get('english', ' '.join(words)),
            res.get('hindi',   ''),
            res.get('marathi', '')
        )
    except json.JSONDecodeError:
        print(f"⚠️  JSON parse error")
        return ' '.join(words), '', ''
    except Exception as e:
        print(f"⚠️  Groq error: {e}")
        return ' '.join(words), '', ''

# ── Devanagari font ───────────────────────────────────────────
FONT_PATH = 'NotoSansDevanagari.ttf'
try:
    deva_font_lg = ImageFont.truetype(FONT_PATH, 22)
    deva_font_sm = ImageFont.truetype(FONT_PATH, 16)
    print("✅ Devanagari font loaded")
except:
    deva_font_lg = deva_font_sm = None
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

def wrap_text_pil(text, font, max_width):
    if not font or not text:
        return [text]
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = (line + " " + word).strip()
        w    = font.getbbox(test)[2]
        if w <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

def wrap_text_cv(text, max_width, font, scale, thickness):
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        tw   = cv2.getTextSize(test, font, scale, thickness)[0][0]
        if tw < max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    lines.append(line)
    return lines

# ── UI helpers ────────────────────────────────────────────────
def draw_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.85):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+r,y1), (x2-r,y2), color, -1)
    cv2.rectangle(overlay, (x1,y1+r), (x2,y2-r), color, -1)
    cv2.circle(overlay, (x1+r,y1+r), r, color, -1)
    cv2.circle(overlay, (x2-r,y1+r), r, color, -1)
    cv2.circle(overlay, (x1+r,y2-r), r, color, -1)
    cv2.circle(overlay, (x2-r,y2-r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

# ── Colors (BGR) ──────────────────────────────────────────────
C_BG      = (18,  18,  24 )
C_PANEL   = (28,  28,  38 )
C_CARD    = (40,  40,  55 )
C_ACCENT  = (0,   165, 255)   # orange
C_GREEN   = (80,  220, 120)
C_YELLOW  = (80,  220, 255)
C_RED     = (80,  80,  220)
C_TEXT    = (230, 230, 240)
C_SUBTEXT = (130, 130, 155)
C_HI      = (120, 220, 140)
C_MR      = (140, 180, 255)
C_EN      = (255, 200, 100)
C_BORDER  = (55,  55,  75 )
C_VOTE_OK = (80,  200, 100)
C_VOTE_NO = (50,  50,  70 )

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ── Load artifacts ───────────────────────────────────────────
print("\n" + "="*55)
print("   ISL Detector + Groq NLP + Hindi / Marathi")
print("="*55)

print("\n[1/4] Loading model...")
try:
    model = tf.keras.models.load_model('best_model.keras')
    print(f"      ✅ input: {model.input_shape}")
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
    FEAT = mean.shape[-1]
    print(f"      ✅ features: {FEAT}")
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
        pts = {i:(int(lm.x*w),int(lm.y*h))
               for i,lm in enumerate(result.pose_landmarks)}
        for a,b in POSE_CONN:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0,180,180), 2)
        for p in pts.values():
            cv2.circle(frame, p, 3, (0,255,255), -1)
    if result.left_hand_landmarks:
        pts = {i:(int(lm.x*w),int(lm.y*h))
               for i,lm in enumerate(result.left_hand_landmarks)}
        for a,b in HAND_CONN:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (180,60,0), 2)
        for p in pts.values():
            cv2.circle(frame, p, 4, (255,120,0), -1)
    if result.right_hand_landmarks:
        pts = {i:(int(lm.x*w),int(lm.y*h))
               for i,lm in enumerate(result.right_hand_landmarks)}
        for a,b in HAND_CONN:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0,60,180), 2)
        for p in pts.values():
            cv2.circle(frame, p, 4, (0,120,255), -1)

def extract_keypoints(result):
    pose = np.array([[lm.x,lm.y,lm.z]
                     for lm in result.pose_landmarks]).flatten() \
           if result.pose_landmarks else np.zeros(99)
    lh   = np.array([[lm.x,lm.y,lm.z]
                     for lm in result.left_hand_landmarks]).flatten() \
           if result.left_hand_landmarks else np.zeros(63)
    rh   = np.array([[lm.x,lm.y,lm.z]
                     for lm in result.right_hand_landmarks]).flatten() \
           if result.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# ── State ─────────────────────────────────────────────────────
MAX_FRAMES        = 30
PREDICT_EVERY     = 8
CONFIDENCE_THRESH = 0.50
WORD_COOLDOWN     = 25
VOTE_WINDOW       = 5
VOTE_THRESHOLD    = 3
PANEL_W           = 420

BUFFER           = deque(maxlen=MAX_FRAMES)
detected_words   = []
sentence_en      = ""
sentence_hi      = ""
sentence_mr      = ""
prediction       = "Warming up..."
confidence       = 0.0
top3             = []
word_vote_buffer = []
frame_count      = 0
last_word_frame  = 0
last_word        = ""
fps_time         = time.time()
fps              = 0
nlp_processing   = False
status_msg       = ""
status_time      = 0

def set_status(msg):
    global status_msg, status_time
    status_msg  = msg
    status_time = time.time()

def run_nlp(words):
    global sentence_en, sentence_hi, sentence_mr, nlp_processing
    nlp_processing = True
    set_status("Groq is thinking...")
    print(f"\n🧠 Sending to Groq: {[str(w) for w in words]}")
    en, hi, mr  = call_groq([str(w) for w in words])
    sentence_en = en
    sentence_hi = normalize_indic(hi, 'hi') if hi else ''
    sentence_mr = normalize_indic(mr, 'mr') if mr else ''
    print(f"   EN : {sentence_en}")
    print(f"   HI : {sentence_hi}")
    print(f"   MR : {sentence_mr}")
    nlp_processing = False
    set_status("✓ Done!")

# ── Webcam ───────────────────────────────────────────────────
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found!"); sys.exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1380)
print("✅ Ready\n")
print("SPACE=Process | Z=Undo | 1/2/3=Remove | C=Clear | Q=Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cam_h, cam_w = frame.shape[:2]
    frame_count += 1

    # ── MediaPipe ────────────────────────────────────────────
    try:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)
        draw_landmarks(frame, result, cam_h, cam_w)
        kp = extract_keypoints(result)
        BUFFER.append(kp)
    except:
        BUFFER.append(np.zeros(FEAT))

    # ── Prediction + voting ──────────────────────────────────
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

            # Voting
            if confidence >= CONFIDENCE_THRESH:
                word_vote_buffer.append(str(prediction))
            else:
                word_vote_buffer.append(None)
            if len(word_vote_buffer) > VOTE_WINDOW:
                word_vote_buffer.pop(0)

            if len(word_vote_buffer) == VOTE_WINDOW:
                valid = [w for w in word_vote_buffer if w is not None]
                if valid:
                    best_word, count = Counter(valid).most_common(1)[0]
                    if (count >= VOTE_THRESHOLD and
                            best_word != str(last_word) and
                            frame_count - last_word_frame > WORD_COOLDOWN and
                            len(detected_words) < 8):
                        detected_words.append(best_word)
                        last_word        = best_word
                        last_word_frame  = frame_count
                        word_vote_buffer.clear()
                        set_status(f"+ '{best_word}' added!")
                        print(f"✅ Word: '{best_word}' ({count}/{VOTE_WINDOW} votes)")
        except Exception as e:
            print(f"Pred error: {e}")

    # ── FPS ──────────────────────────────────────────────────
    if frame_count % 30 == 0:
        fps      = 30 / (time.time() - fps_time)
        fps_time = time.time()

    # ════════════════════════════════════════════════════════
    # BUILD CANVAS
    # ════════════════════════════════════════════════════════
    total_w = cam_w + PANEL_W
    canvas  = np.full((cam_h, total_w, 3), C_BG, dtype=np.uint8)
    canvas[:, :cam_w] = frame

    # ── Camera overlays ──────────────────────────────────────

    # Top prediction bar on camera
    draw_rect(canvas, 0, 0, cam_w, 48, (8,8,12), alpha=0.82)
    color = C_GREEN  if confidence > 0.65 else \
            C_YELLOW if confidence > 0.40 else C_RED
    cv2.putText(canvas, str(prediction).upper(),
                (10, 33), FONT_BOLD, 1.0, color, 2)
    cv2.putText(canvas, f"{confidence*100:.0f}%",
                (cam_w-70, 33), FONT, 0.85, (255,220,60), 2)

    # Buffer bar
    filled = int((len(BUFFER)/MAX_FRAMES) * cam_w)
    cv2.rectangle(canvas, (0, cam_h-7), (cam_w, cam_h), (25,25,35), -1)
    cv2.rectangle(canvas, (0, cam_h-7), (filled, cam_h), C_GREEN, -1)

    # FPS
    cv2.putText(canvas, f"FPS {fps:.0f}",
                (8, cam_h-12), FONT, 0.36, C_SUBTEXT, 1)

    # Vote dots bottom right of camera
    vx = cam_w - 140
    cv2.putText(canvas, "VOTES", (vx, cam_h-12),
                FONT, 0.34, C_SUBTEXT, 1)
    for i, vote in enumerate(word_vote_buffer[-VOTE_WINDOW:]):
        cx   = vx + 52 + i * 18
        vcol = C_VOTE_OK if vote == str(prediction) else \
               C_ACCENT  if (vote and vote != str(prediction)) else \
               C_VOTE_NO
        cv2.circle(canvas, (cx, cam_h-14), 6, vcol, -1)
        cv2.circle(canvas, (cx, cam_h-14), 6, C_BORDER, 1)

    # ── Panel background ─────────────────────────────────────
    px = cam_w
    canvas[:, px:] = C_PANEL
    cv2.line(canvas, (px,0), (px, cam_h), C_BORDER, 2)

    py = 16

    # Header
    cv2.putText(canvas, "ISL", (px+12, py+24),
                FONT_BOLD, 1.1, C_ACCENT, 2)
    cv2.putText(canvas, "DETECTOR", (px+70, py+24),
                FONT_BOLD, 1.1, C_TEXT, 2)
    cv2.putText(canvas, "Sign Language  →  Text  →  Translation",
                (px+10, py+42), FONT, 0.32, C_SUBTEXT, 1)
    py += 58
    cv2.line(canvas, (px+8,py), (total_w-8,py), C_BORDER, 1)
    py += 14

    # ── Top 3 ────────────────────────────────────────────────
    cv2.putText(canvas, "TOP PREDICTIONS", (px+10, py),
                FONT, 0.37, C_SUBTEXT, 1)
    py += 16

    for i, (word, prob) in enumerate(top3 if top3 else [('—',0)]*3):
        bx1  = px + 10
        bx2  = total_w - 10
        barx = bx1 + int((bx2-bx1) * prob)
        bcol = C_GREEN  if i==0 else \
               C_ACCENT if i==1 else C_SUBTEXT

        cv2.rectangle(canvas, (bx1,py), (bx2,py+20), C_CARD, -1)
        if prob > 0:
            cv2.rectangle(canvas, (bx1,py), (barx,py+20),
                          tuple(max(0,int(c*0.45)) for c in bcol), -1)
        cv2.rectangle(canvas, (bx1,py), (bx2,py+20), C_BORDER, 1)
        cv2.putText(canvas, f"  {str(word):<14}{prob*100:.0f}%",
                    (bx1+4, py+14), FONT, 0.4, bcol, 1)
        py += 25

    py += 6
    cv2.line(canvas, (px+8,py), (total_w-8,py), C_BORDER, 1)
    py += 14

    # ── Detected words ───────────────────────────────────────
    cv2.putText(canvas, "DETECTED WORDS", (px+10, py),
                FONT, 0.37, C_SUBTEXT, 1)
    py += 18

    if detected_words:
        x_off = px + 10
        row_y = py
        for idx, word in enumerate(detected_words):
            label = f"{idx+1}. {str(word)}"
            tw    = cv2.getTextSize(label, FONT, 0.4, 1)[0][0]
            bw    = tw + 14
            if x_off + bw > total_w - 10:
                x_off = px + 10
                row_y += 26
            draw_rounded_rect(canvas, x_off, row_y-15,
                              x_off+bw, row_y+7, 5, (55,75,110))
            cv2.rectangle(canvas, (x_off, row_y-15),
                          (x_off+bw, row_y+7), (75,105,160), 1)
            cv2.putText(canvas, label, (x_off+7, row_y),
                        FONT, 0.4, C_TEXT, 1)
            x_off += bw + 8
        py = row_y + 30
    else:
        cv2.putText(canvas, "Start signing to detect words...",
                    (px+10, py+14), FONT, 0.37, C_SUBTEXT, 1)
        py += 30

    py += 4
    cv2.line(canvas, (px+8,py), (total_w-8,py), C_BORDER, 1)
    py += 14

    # ── English ──────────────────────────────────────────────
    # Label pill
    draw_rounded_rect(canvas, px+8, py-2, px+80, py+16, 4, (80,55,10))
    cv2.putText(canvas, "ENGLISH", (px+13, py+12),
                FONT, 0.36, C_EN, 1)
    py += 24

    en_lines = wrap_text_cv(
        sentence_en if sentence_en else "Press SPACE to generate...",
        PANEL_W-22, FONT, 0.4, 1)
    for ln in en_lines[:3]:
        col = C_TEXT if sentence_en else C_SUBTEXT
        cv2.putText(canvas, ln, (px+10, py), FONT, 0.4, col, 1)
        py += 18
    py += 6
    cv2.line(canvas, (px+8,py), (total_w-8,py), C_BORDER, 1)
    py += 12

    # ── Hindi ─────────────────────────────────────────────────
    draw_rounded_rect(canvas, px+8, py-2, px+65, py+16, 4, (15,65,20))
    cv2.putText(canvas, "HINDI", (px+13, py+12),
                FONT, 0.36, C_HI, 1)
    py += 24

    if sentence_hi:
        hi_lines = wrap_text_pil(sentence_hi, deva_font_lg, PANEL_W-22)
        for ln in hi_lines[:3]:
            canvas = put_devanagari(canvas, ln, (px+10, py-4),
                                    deva_font_lg, C_TEXT)
            py += 26
    else:
        cv2.putText(canvas, "—", (px+10, py),
                    FONT, 0.4, C_SUBTEXT, 1)
        py += 20

    py += 4
    cv2.line(canvas, (px+8,py), (total_w-8,py), C_BORDER, 1)
    py += 12

    # ── Marathi ───────────────────────────────────────────────
    draw_rounded_rect(canvas, px+8, py-2, px+78, py+16, 4, (15,25,70))
    cv2.putText(canvas, "MARATHI", (px+13, py+12),
                FONT, 0.36, C_MR, 1)
    py += 24

    if sentence_mr:
        mr_lines = wrap_text_pil(sentence_mr, deva_font_lg, PANEL_W-22)
        for ln in mr_lines[:3]:
            canvas = put_devanagari(canvas, ln, (px+10, py-4),
                                    deva_font_lg, C_TEXT)
            py += 26
    else:
        cv2.putText(canvas, "—", (px+10, py),
                    FONT, 0.4, C_SUBTEXT, 1)
        py += 20

    py += 4
    cv2.line(canvas, (px+8,py), (total_w-8,py), C_BORDER, 1)

    # ── Status message ───────────────────────────────────────
    if status_msg and time.time() - status_time < 3:
        alpha  = min(1.0, 3.0 - (time.time() - status_time))
        scol   = C_GREEN  if ("✓" in status_msg or "+" in status_msg) else \
                 C_ACCENT if "⚠" in status_msg else \
                 (80, 200, 255)
        cv2.putText(canvas, status_msg, (px+10, py+22),
                    FONT, 0.42, scol, 1)

    # Groq processing animation
    if nlp_processing:
        dots = "." * (int(time.time()*3) % 4)
        cv2.putText(canvas, f"Groq thinking{dots}",
                    (px+10, py+40), FONT, 0.42, C_ACCENT, 1)

    # ── Controls ─────────────────────────────────────────────
    ctrl_y = cam_h - 88
    cv2.line(canvas, (px+8, ctrl_y-6), (total_w-8, ctrl_y-6), C_BORDER, 1)
    cv2.putText(canvas, "CONTROLS",
                (px+10, ctrl_y+10), FONT, 0.35, C_SUBTEXT, 1)

    ctrls = [
        ("SPACE", "Process → Groq NLP"),
        ("Z",     "Undo last word"),
        ("1/2/3", "Remove word by number"),
        ("C",     "Clear everything"),
        ("Q",     "Quit"),
    ]
    for i, (k, d) in enumerate(ctrls):
        ky = ctrl_y + 24 + i*13
        cv2.putText(canvas, k, (px+10, ky), FONT, 0.33, C_ACCENT,  1)
        cv2.putText(canvas, d, (px+72, ky), FONT, 0.33, C_SUBTEXT, 1)

    # Landmark legend
    leg_items = [(( 0,255,255),"Pose"),((255,120,0),"L.Hand"),((0,120,255),"R.Hand")]
    lx = px + 10
    for col, lbl in leg_items:
        cv2.circle(canvas, (lx+4, cam_h-8), 4, col, -1)
        cv2.putText(canvas, lbl, (lx+12, cam_h-4),
                    FONT, 0.28, C_SUBTEXT, 1)
        lx += 90

    # ── Show ─────────────────────────────────────────────────
    cv2.imshow('ISL Detector + Groq NLP', canvas)

    # ── Keys ─────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        detected_words.clear()
        sentence_en = sentence_hi = sentence_mr = ""
        last_word   = ""
        word_vote_buffer.clear()
        BUFFER.clear()
        set_status("Cleared!")
        print("🔄 Cleared")

    elif key == ord('z'):
        if detected_words:
            removed   = detected_words.pop()
            last_word = detected_words[-1] if detected_words else ""
            word_vote_buffer.clear()
            set_status(f"↩ Removed '{removed}'")
            print(f"↩️  Undo: '{removed}'")

    elif key == ord(' '):
        if detected_words and not nlp_processing:
            t = threading.Thread(
                target=run_nlp,
                args=(detected_words.copy(),))
            t.daemon = True
            t.start()
        elif not detected_words:
            set_status("⚠ No words detected yet!")

    elif key == ord('1') and len(detected_words) >= 1:
        removed = detected_words.pop(0)
        set_status(f"Removed: '{removed}'")
        print(f"🗑️ Removed word 1: '{removed}'")

    elif key == ord('2') and len(detected_words) >= 2:
        removed = detected_words.pop(1)
        set_status(f"Removed: '{removed}'")
        print(f"🗑️ Removed word 2: '{removed}'")

    elif key == ord('3') and len(detected_words) >= 3:
        removed = detected_words.pop(2)
        set_status(f"Removed: '{removed}'")
        print(f"🗑️ Removed word 3: '{removed}'")

landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("\n👋 ISL Detector closed.")