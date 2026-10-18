import argparse, os
import cv2, numpy as np, tensorflow as tf
from mtcnn import MTCNN

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--model", default="outputs/v2/mixed_cons.keras")
p.add_argument("--threshold", type=float, default=0.3446)  # c23 val threshold of mixed_cons
p.add_argument("--frames", type=int, default=16)
a = p.parse_args()

# Crop settings copied from the training pipeline: MTCNN conf>=0.85, largest face,
# 30% margin, square, reflect padding, 299x299 Lanczos.
MIN_CONF, MARGIN, SIZE = 0.85, 1.30, 299

cap = cv2.VideoCapture(a.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if n <= 0:
    raise SystemExit("Cannot read video: " + a.video)
idxs = np.linspace(0, n - 1, a.frames).astype(int)

detector = MTCNN()
tmp = os.path.join("outputs", "v2", "predict_tmp", os.path.splitext(os.path.basename(a.video))[0])
os.makedirs(tmp, exist_ok=True)

def crop(img_bgr):
    faces = detector.detect_faces(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    faces = [f for f in faces if f["confidence"] >= MIN_CONF and f["box"][2] > 10 and f["box"][3] > 10]
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda f: f["box"][2] * f["box"][3])["box"]
    H, W = img_bgr.shape[:2]
    half = max(w, h) * MARGIN / 2.0
    cx, cy = x + w / 2.0, y + h / 2.0
    x1, y1, x2, y2 = [int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half)]
    pl, pt, pr, pb = max(0, -x1), max(0, -y1), max(0, x2 - W), max(0, y2 - H)
    if pl or pt or pr or pb:
        img_bgr = cv2.copyMakeBorder(img_bgr, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        c = img_bgr[y1 + pt:y2 + pt, x1 + pl:x2 + pl]
    else:
        c = img_bgr[y1:y2, x1:x2]
    c = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)
    # JPEG round trip, because training crops were stored as JPEG files
    ok, buf = cv2.imencode(".jpg", c)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

crops = []
for k, i in enumerate(idxs):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, fr = cap.read()
    if not ok:
        continue
    c = crop(fr)
    if c is not None:
        cv2.imwrite(os.path.join(tmp, f"crop_{k:02d}.jpg"), c)
        crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Frames sampled: {len(idxs)} | faces found: {len(crops)} | crops saved in {tmp}")
if len(crops) < 8:
    print("WARNING: fewer than 8 usable face crops. The result is unreliable.")
if not crops:
    raise SystemExit("No face detected. Cannot classify.")

model = tf.keras.models.load_model(a.model, compile=False, safe_mode=False)
probs = model.predict(np.stack(crops).astype("float32"), verbose=0).ravel()  # raw 0-255 RGB; preprocessing is inside the model
mean_p = float(probs.mean())

print("Per-frame fake probability:", np.round(probs, 3).tolist())
print(f"Mean fake probability: {mean_p:.3f} (threshold {a.threshold})")
print("Verdict:", "LIKELY FAKE" if mean_p >= a.threshold else "LIKELY REAL")
print("This is a probability from a model trained only on FF++ Deepfakes, not a proof.")