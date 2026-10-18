import argparse, csv, glob, json, os
import numpy as np, tensorflow as tf
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score, confusion_matrix)

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--model", required=True)
p.add_argument("--root", default="frames_faces")   # frames_compressed for the OLD full-frame models
p.add_argument("--variant", required=True)          # test variant: c23 / c40 / cascade
p.add_argument("--thr-variant", default=None)       # variant whose VAL set picks the threshold (default = --variant)
p.add_argument("--n-boot", type=int, default=1000)
a = p.parse_args()
thr_variant = a.thr_variant or a.variant
OUT = "outputs/v2/eval"; os.makedirs(OUT, exist_ok=True)

model = tf.keras.models.load_model(a.model, compile=False, safe_mode=False)

def load(path):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.resize(img, (299, 299))

def video_probs(split, variant):
    items = []
    for cls, lbl in [("real", 0), ("fake", 1)]:
        for f in sorted(glob.glob(os.path.join(a.root, variant, split, cls, "*", "*.jpg"))):
            items.append((f, lbl, cls + "/" + os.path.basename(os.path.dirname(f))))
    if not items:
        raise SystemExit(f"No frames under {a.root}/{variant}/{split}")
    ds = (tf.data.Dataset.from_tensor_slices([i[0] for i in items])
          .map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(2))
    pr = model.predict(ds, verbose=0).ravel()
    vids = {}
    for (f, l, v), q in zip(items, pr):
        vids.setdefault(v, {"label": l, "p": []})["p"].append(float(q))
    names = sorted(vids)
    return (names, np.array([vids[n]["label"] for n in names]),
            np.array([np.mean(vids[n]["p"]) for n in names]),
            np.array([len(vids[n]["p"]) for n in names]))

def metrics(y, pr, thr):
    pred = (pr >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float((pred == y).mean()),
            "roc_auc": float(roc_auc_score(y, pr)) if len(set(y)) > 1 else float("nan"),
            "precision_fake": float(precision_score(y, pred, zero_division=0)),
            "recall_fake": float(recall_score(y, pred, zero_division=0)),
            "f1_fake": float(f1_score(y, pred, zero_division=0)),
            "recall_real": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

# 1) threshold from VAL only (Youden's J)
_, yv, pv, _ = video_probs("val", thr_variant)
fpr, tpr, th = roc_curve(yv, pv)
thr = float(th[np.argmax(tpr - fpr)])
print(f"Threshold {thr:.4f} from val ({thr_variant}, {len(yv)} videos)")

# 2) TEST, video level
names, yt, pt, nfr = video_probs("test", a.variant)
res = metrics(yt, pt, thr)
print(f"TEST {a.variant}: {len(yt)} videos, frames/video {nfr.min()}-{nfr.max()}")

# 3) bootstrap over videos
rng = np.random.default_rng(42)
boots = {k: [] for k in ["accuracy", "roc_auc", "precision_fake", "recall_fake", "f1_fake"]}
for _ in range(a.n_boot):
    idx = rng.integers(0, len(yt), len(yt))
    if len(set(yt[idx])) < 2:
        continue
    m = metrics(yt[idx], pt[idx], thr)
    for k in boots:
        boots[k].append(m[k])
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}

out = {"name": a.name, "model": a.model, "root": a.root, "test_variant": a.variant,
       "threshold_variant": thr_variant, "threshold": thr, "n_test_videos": int(len(yt)),
       "aggregation": "mean frame probability per video",
       "ci_method": f"percentile bootstrap over test videos, {a.n_boot} resamples, seed 42, 95%",
       "test": res, "ci95": ci}
tag = f"{a.name}_{a.variant}" + (f"_thr{thr_variant}" if thr_variant != a.variant else "")
json.dump(out, open(f"{OUT}/{tag}.json", "w"), indent=2)
np.savetxt(f"{OUT}/{tag}_video_probs.csv", np.c_[yt, pt], delimiter=",", header="label,prob", comments="")

row = [a.name, a.variant, thr_variant, len(yt)] + \
      [f"{res[k]:.3f} [{ci[k][0]:.3f},{ci[k][1]:.3f}]" for k in ["accuracy", "roc_auc", "precision_fake", "recall_fake", "f1_fake"]] + \
      [res["tn"], res["fp"], res["fn"], res["tp"]]
sp = f"{OUT}/summary.csv"
new = not os.path.exists(sp)
with open(sp, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["name", "test_variant", "thr_variant", "n_videos", "acc", "auc", "prec_fake", "rec_fake", "f1_fake", "TN", "FP", "FN", "TP"])
    w.writerow(row)
print(json.dumps({"test": res, "ci95": ci}, indent=2))