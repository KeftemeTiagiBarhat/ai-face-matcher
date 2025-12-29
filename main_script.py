
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#CONFIG
KNOWN_DIR = "known_faces"
os.makedirs(KNOWN_DIR, exist_ok=True)

CAP_DEVICE = 0
CAP_W, CAP_H = 640, 480
DETECTION_SCALE = 0.6        # scale for detection
DETECT_INTERVAL = 5         # run detector every N frames
STABLE_FRAMES = 8
DIST_THRESHOLD = 0.6        # tuneable
DF_MODEL = "Facenet512"     # or "VGG-Face", "ArcFace"
EMOTION_INTERVAL = 0.25
MAX_WORKERS = 2
FACE_MIN_SIZE = 60

#HELP
def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else (float(o) if isinstance(o, np.floating) else str(o)))

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o

def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

#Detector (Haar, reliable)
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    raise RuntimeError("Haar cascade not found.")

#Known embeddings cache
known_lock = threading.Lock()
known_embeddings = {}  # id -> np.array
known_meta = {}        # id -> dict

def preload_known_embeddings():
    with known_lock:
        known_embeddings.clear()
        known_meta.clear()
        for pid in os.listdir(KNOWN_DIR):
            pdir = os.path.join(KNOWN_DIR, pid)
            if not os.path.isdir(pdir): continue
            emb_path = os.path.join(pdir, "embedding.json")
            meta_path = os.path.join(pdir, "data.json")
            emb = None
            if os.path.exists(emb_path):
                try:
                    arr = load_json(emb_path)
                    emb = np.array(arr, dtype=np.float32)
                except Exception:
                    emb = None
            else:
                face_j = os.path.join(pdir, "face.jpg")
                if os.path.exists(face_j):
                    img = cv2.imread(face_j)
                    try:
                        rep = DeepFace.represent(img, model_name=DF_MODEL, enforce_detection=False)
                        if isinstance(rep, list) and len(rep)>0:
                            r0 = rep[0]
                            emb = np.array(r0["embedding"] if isinstance(r0, dict) and "embedding" in r0 else r0, dtype=np.float32)
                            save_json(emb_path, emb.tolist())
                    except Exception:
                        emb = None
            if emb is not None:
                known_embeddings[pid] = emb
            meta = load_json(meta_path) if os.path.exists(meta_path) else {"name": pid}
            known_meta[pid] = meta
    print(f"[INFO] Loaded {len(known_embeddings)} known faces.")

def find_best_match_by_embedding(emb):
    if emb is None:
        return None, None, None
    best_id, best_meta, best_dist = None, None, float("inf")
    with known_lock:
        for pid, p_emb in known_embeddings.items():
            try:
                d = float(np.linalg.norm(emb - p_emb))
            except Exception:
                continue
            if d < best_dist:
                best_dist = d
                best_id = pid
                best_meta = known_meta.get(pid)
    if best_id and best_dist <= DIST_THRESHOLD:
        return best_id, best_meta, best_dist
    return None, None, best_dist

#Thread pool
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

#Background tasks
def process_stable_face(best_frame_bgr, faces_store, fid):
    """Compute embedding, compare with DB; if not found -> analyze and save"""
    faces_store[fid]['status'] = "processing"
    try:
        small = cv2.resize(best_frame_bgr, (160,160))
        rep = DeepFace.represent(small, model_name=DF_MODEL, enforce_detection=False)
        emb = None
        if isinstance(rep, list) and len(rep)>0:
            r0 = rep[0]
            emb = np.array(r0["embedding"] if isinstance(r0, dict) and "embedding" in r0 else r0, dtype=np.float32)
        elif isinstance(rep, dict) and "embedding" in rep:
            emb = np.array(rep["embedding"], dtype=np.float32)
        faces_store[fid]['embedding'] = emb.tolist() if emb is not None else None

        pid, meta, dist = find_best_match_by_embedding(emb)
        if pid:
            faces_store[fid]['analysis'] = meta
            faces_store[fid]['matched'] = True
            faces_store[fid]['status'] = f"known (d={dist:.3f})"
            return

        #not found -> analyze
        faces_store[fid]['status'] = "analyzing"
        rgb = cv2.cvtColor(best_frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            res = DeepFace.analyze(rgb, actions=["age","gender","race"], enforce_detection=False)
            if isinstance(res, list): res = res[0]
        except Exception:
            res = {}

        age = res.get("age")
        gender = None
        race = None
        if isinstance(res.get("gender"), dict):
            gender = max(res["gender"], key=res["gender"].get)
        else:
            gender = res.get("gender")
        if isinstance(res.get("race"), dict):
            race = max(res["race"], key=res["race"].get)
        else:
            race = res.get("race")

        #save person
        new_id = str(uuid.uuid4())
        pdir = os.path.join(KNOWN_DIR, new_id)
        os.makedirs(pdir, exist_ok=True)
        face_path = os.path.join(pdir, "face.jpg")
        emb_path = os.path.join(pdir, "embedding.json")
        data_path = os.path.join(pdir, "data.json")
        cv2.imwrite(face_path, best_frame_bgr)

        if emb is not None:
            save_json(emb_path, emb.tolist())
            with known_lock:
                known_embeddings[new_id] = emb
                known_meta[new_id] = {"name":"unknown","age":age,"gender":gender,"race":race,"face_path":face_path}

        save_json(data_path, {"name":"unknown","age":age,"gender":gender,"race":race,"face_path":face_path})

        faces_store[fid]['analysis'] = {"name":"unknown","age":age,"gender":gender,"race":race,"face_path":face_path}
        faces_store[fid]['status'] = "saved"
    except Exception as e:
        faces_store[fid]['status'] = "error"
        print("process_stable_face:", e)

def emotion_worker(fid, face_ref, faces_store):
    last = 0.0
    while True:
        if fid not in face_ref:
            time.sleep(0.05); continue
        now = time.time()
        if now - last < EMOTION_INTERVAL:
            time.sleep(0.01); continue
        img = face_ref.get(fid)
        if img is None:
            time.sleep(0.05); continue
        try:
            small = cv2.resize(img, (160,160))
            res = DeepFace.analyze(small, actions=["emotion"], enforce_detection=False)
            if isinstance(res, list): res = res[0]
            emo = None
            if isinstance(res, dict):
                emo = res.get("dominant_emotion") or (max(res.get("emotion",{}), key=res.get("emotion",{}).get) if res.get("emotion") else None)
            faces_store[fid]['emotion'] = emo or ""
        except Exception:
            faces_store[fid]['emotion'] = ""
        last = time.time()

#Main
def main():
    preload_known_embeddings()
    cap = cv2.VideoCapture(CAP_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    if not cap.isOpened():
        print("Cannot open camera"); return

    faces_store = {}  # fid -> dict
    face_ref = {}     # fid -> latest crop
    last_box = {}     # fid -> last box (x,y,w,h)
    frame_idx = 0
    print("Start. Press 'q' to quit.")

    #parameters for centroid matching
    CENTER_MATCH_DIST = 2000  #squared pixels threshold, tuneable

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue
        frame_idx += 1
        small_frame = cv2.resize(frame, (0,0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)

        boxes = []
        if frame_idx % DETECT_INTERVAL == 0:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE))
            if len(dets):
                # scale back
                boxes = [(int(x/DETECTION_SCALE), int(y/DETECTION_SCALE), int(w/DETECTION_SCALE), int(h/DETECTION_SCALE)) for (x,y,w,h) in dets]

            #process each detected face
            for (x,y,w,h) in boxes:
                cx, cy = x + w//2, y + h//2
                #match to existing face by centroid
                fid = None; bestd = CENTER_MATCH_DIST
                for f, info in last_box.items():
                    lx, ly, lw, lh = info
                    dc = (lx+lw//2 - cx)**2 + (ly+lh//2 - cy)**2
                    size_diff = abs(lw - w)
                    if dc < bestd and size_diff < max(80, w//2):
                        bestd = dc; fid = f
                if fid is None:
                    #create new id
                    fid = str(uuid.uuid4())
                    faces_store[fid] = {
                        "frames": deque(maxlen=STABLE_FRAMES),
                        "best_frame": None,
                        "best_score": -1.0,
                        "analyzed": False,
                        "analysis": None,
                        "embedding": None,
                        "emotion": "",
                        "matched": False,
                        "status": "waiting"
                    }
                    face_ref[fid] = None
                    threading.Thread(target=emotion_worker, args=(fid, face_ref, faces_store), daemon=True).start()

                last_box[fid] = (x,y,w,h)
                #crop and update
                x2,y2 = x+w, y+h
                crop = frame[y:y2, x:x2].copy()
                if crop.size==0: continue
                faces_store[fid]['frames'].append(crop)
                face_ref[fid] = crop.copy()

                # quality score
                try:
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharp = laplacian_var(gray_crop)
                    rel_size = (w*h) / float(frame.shape[0]*frame.shape[1])
                    score = (np.sqrt(max(sharp,0))*0.6) + (rel_size*100.0)
                    if score > faces_store[fid]['best_score']:
                        faces_store[fid]['best_score'] = score
                        faces_store[fid]['best_frame'] = crop.copy()
                except Exception:
                    pass

                # schedule stable processing
                if (not faces_store[fid]['analyzed']) and (len(faces_store[fid]['frames']) >= STABLE_FRAMES):
                    faces_store[fid]['analyzed'] = True
                    best = faces_store[fid]['best_frame'] if faces_store[fid]['best_frame'] is not None else faces_store[fid]['frames'][-1]
                    faces_store[fid]['status'] = "queued"
                    executor.submit(process_stable_face, best, faces_store, fid)

        #draw rectangles for all tracked faces (use last_box)
        now = time.time()
        stale = []
        for fid, box in list(last_box.items()):
            x,y,w,h = box
            #if face not updated recently, remove
            #(we don't track timestamps here; rely on frames queue length or TTL if needed)
            info = faces_store.get(fid, {})
            #compose text
            if info.get('analysis') and info.get('matched', False):
                meta = info['analysis']
                name = meta.get('name','unknown')
                age = meta.get('age','')
                gender = meta.get('gender','')
                race = meta.get('race','')
                lines = [f"Name: {name}", f"Age:{age} Gender:{gender}", f"Race:{race}", f"Emo:{info.get('emotion','')}"]
            elif info.get('analysis'):
                a = info['analysis']
                lines = [f"{a.get('gender','')}, {a.get('race','')}", f"Age:{a.get('age','')}", f"Emo:{info.get('emotion','')}", f"Status:{info.get('status','')}"]
            else:
                lines = [f"Status:{info.get('status','')}", f"Emo:{info.get('emotion','')}"]

            #outer rect
            outer_x1 = max(0, x-8)
            outer_y1 = max(0, y-8- (len(lines)*16))
            outer_x2 = min(frame.shape[1]-1, x+w+8)
            outer_y2 = min(frame.shape[0]-1, y+h+8)
            cv2.rectangle(frame, (outer_x1, outer_y1), (outer_x2, outer_y2), (0,255,0), 2)

            #info box
            text_x = outer_x1 + 6
            text_y = outer_y1 + 16
            for ln in lines:
                cv2.putText(frame, ln, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
                text_y += 16
            #face rect
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,200,0), 2)

        cv2.imshow("AI FaceMatcher", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
