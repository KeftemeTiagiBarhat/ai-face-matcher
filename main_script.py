import os
import cv2
import time
import json
import uuid
import threading
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from deepface import DeepFace
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#CONFIG
CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 480
FACE_SIZE = 160
FRAMES_FOR_EMB = 5
DIST_THRESHOLD = 0.6
MAX_WORKERS = 2
EMOTION_INTERVAL = 0.3
DB_PATH = "face_db.json"

#UTILS
def cosine(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

#MODELS
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.6
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
DB = load_db()
faces = {}
face_id_counter = 0

#WORKERS
def compute_embedding(face):
    rep = DeepFace.represent(
        face, model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False
    )
    return np.array(rep[0]["embedding"])

def analyze_face(face):
    r = DeepFace.analyze(
        face,
        actions=["age", "gender", "race"],
        detector_backend="skip",
        enforce_detection=False
    )[0]
    return {
        "age": r["age"],
        "gender": r["dominant_gender"],
        "race": r["dominant_race"]
    }

def emotion_worker(fid):
    last = 0
    while fid in faces:
        now = time.time()
        if now - last < EMOTION_INTERVAL:
            time.sleep(0.05)
            continue
        face = faces[fid].get("last_face")
        if face is not None:
            try:
                r = DeepFace.analyze(
                    face,
                    actions=["emotion"],
                    detector_backend="skip",
                    enforce_detection=False
                )[0]
                faces[fid]["emotion"] = r["dominant_emotion"]
            except:
                pass
        last = now

def process_face(fid):
    f = faces[fid]
    best = max(f["frames"], key=lambda img: laplacian_var(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    emb = compute_embedding(best)

    for person in DB:
        if cosine(emb, np.array(person["embedding"])) < DIST_THRESHOLD:
            f["info"] = person["meta"]
            f["status"] = "KNOWN"
            return

    meta = analyze_face(best)
    DB.append({
        "id": str(uuid.uuid4()),
        "embedding": emb.tolist(),
        "meta": meta
    })
    save_db(DB)
    f["info"] = meta
    f["status"] = "NEW"

#MAIN
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("[INFO] System started")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    if res.detections:
        for det in res.detections:
            box = det.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x, y = int(box.xmin * w), int(box.ymin * h)
            bw, bh = int(box.width * w), int(box.height * h)

            cx, cy = x + bw//2, y + bh//2
            matched = None

            for fid, f in faces.items():
                fx, fy = f["center"]
                if np.linalg.norm([fx-cx, fy-cy]) < 60:
                    matched = fid
                    break

            if matched is None:
                matched = face_id_counter
                faces[matched] = {
                    "center": (cx, cy),
                    "frames": deque(maxlen=FRAMES_FOR_EMB),
                    "status": "COLLECT",
                    "info": None,
                    "emotion": ""
                }
                threading.Thread(target=emotion_worker, args=(matched,), daemon=True).start()
                face_id_counter += 1

            faces[matched]["center"] = (cx, cy)

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
            faces[matched]["frames"].append(face)
            faces[matched]["last_face"] = face

            if len(faces[matched]["frames"]) == FRAMES_FOR_EMB and faces[matched]["status"] == "COLLECT":
                faces[matched]["status"] = "PROCESSING"
                executor.submit(process_face, matched)

            label = faces[matched]["status"]
            if faces[matched]["info"]:
                label += f" | {faces[matched]['info']} | {faces[matched]['emotion']}"

            cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

    cv2.imshow("AI FaceMatcher", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()
