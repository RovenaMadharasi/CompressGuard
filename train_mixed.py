import argparse, glob, json, os, time
import numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)                    # mixed / mixed_cons
p.add_argument("--init", default="outputs/v2/2a_c23_head.keras")
p.add_argument("--consistency", type=float, default=0.0)  # lambda; 0 = plain mixed
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--batch", type=int, default=32)           # images per step (pairs use batch//2 pairs)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--patience", type=int, default=4)
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
cons = a.consistency > 0
tf.keras.utils.set_random_seed(42)
ROOT, OUT, VARS = "frames_faces", "outputs/v2", ["c23", "c40", "cascade"]
tag = ("smoke_" if a.smoke else "") + a.tag
ckpt = f"{OUT}/{tag}.keras"

def listing(split):
    P, y, dropped = [], [], 0
    for cls, lbl in [("real", 0), ("fake", 1)]:
        for f in sorted(glob.glob(f"{ROOT}/c23/{split}/{cls}/*/*.jpg")):
            rel = os.path.relpath(f, f"{ROOT}/c23")
            ps = [os.path.join(ROOT, v, rel) for v in VARS]
            if all(os.path.exists(q) for q in ps):
                P.append(ps); y.append(lbl)
            else:
                dropped += 1
    print(f"{split}: {len(y)} frames (real={y.count(0)}, fake={y.count(1)}), dropped (missing in a variant): {dropped}")
    return np.array(P), np.array(y, dtype="float32")

Ptr, ytr = listing("train"); Pva, yva = listing("val")
if a.smoke:
    r = np.random.RandomState(0)
    i = r.permutation(len(ytr))[:96]; Ptr, ytr = Ptr[i], ytr[i]
    i = r.permutation(len(yva))[:64]; Pva, yva = Pva[i], yva[i]; a.epochs = 1

def load(path):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.resize(img, (299, 299))

bs = a.batch // 2 if cons else a.batch
ds = tf.data.Dataset.from_tensor_slices((Ptr, ytr)).shuffle(len(ytr), seed=42, reshuffle_each_iteration=True)
if cons:
    def f(paths, l):
        j = 1 + tf.random.uniform([], 0, 2, tf.int32)      # c40 or cascade
        return load(paths[0]), load(paths[j]), l
else:
    def f(paths, l):
        return load(paths[tf.random.uniform([], 0, 3, tf.int32)]), l
train_ds = ds.map(f, num_parallel_calls=tf.data.AUTOTUNE).batch(bs).prefetch(2)

# VAL: one fixed random variant per frame (seeded), so val covers all three variants at one-variant cost
vi = np.random.RandomState(42).randint(0, 3, len(yva))
val_ds = tf.data.Dataset.from_tensor_slices((Pva[np.arange(len(yva)), vi], yva)) \
    .map(lambda q, l: (load(q), l)).batch(a.batch).prefetch(2)

model = tf.keras.models.load_model(a.init)
base = model.get_layer("xception"); base.trainable = True
for l in base.layers:
    l.trainable = l.name.startswith(("block13", "block14")) and not isinstance(l, layers.BatchNormalization)
bn = [l for l in base.layers if isinstance(l, layers.BatchNormalization)]
print(f"BatchNorm: {len(bn)}, trainable: {sum(l.trainable for l in bn)} (must be 0)")
print("Trainable params:", sum(int(np.prod(w.shape)) for w in model.trainable_weights))
print("Mode:", "mixed + consistency lambda=%g" % a.consistency if cons else "mixed", "| steps/epoch:", len(ytr) // bs)

gap = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D))
fm = tf.keras.Model(model.input, [gap.output, model.output])
opt = tf.keras.optimizers.Adam(a.lr)
bce = tf.keras.losses.BinaryCrossentropy()

@tf.function
def step(xa, xb, y):
    with tf.GradientTape() as t:
        x = tf.concat([xa, xb], 0) if cons else xa
        yy = tf.concat([y, y], 0) if cons else y
        f, pr = fm(x, training=True)
        ce = bce(yy[:, None], pr)
        cl = tf.constant(0.0)
        if cons:
            fa, fb = tf.split(f, 2)
            na, nb = tf.math.l2_normalize(fa, -1), tf.math.l2_normalize(fb, -1)
            cl = tf.reduce_mean(tf.reduce_sum(tf.square(na - nb), -1))
        loss = ce + a.consistency * cl
    g = t.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(g, model.trainable_variables))
    return ce, cl, pr

def evaluate():
    ps, ys = [], []
    for x, yb in val_ds:
        _, pr = fm(x, training=False)
        ps.append(pr.numpy().ravel()); ys.append(yb.numpy())
    ps, ys = np.concatenate(ps), np.concatenate(ys)
    auc = float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else None
    return auc, float(((ps >= 0.5) == ys).mean())

hist, best, bad = [], -1, 0
for ep in range(a.epochs):
    t0, ces, cls_, n = time.time(), 0, 0, 0
    for b in train_ds:
        xa, xb, y = b if cons else (b[0], None, b[1])
        ce, cl, _ = step(xa, xb, y)
        ces += float(ce); cls_ += float(cl); n += 1
        if n % 20 == 0:
            print(f"  ep{ep+1} step {n}  bce={ces/n:.4f}  consist={cls_/n:.4f}", flush=True)
    auc, acc = evaluate()
    rec = {"epoch": ep + 1, "train_bce": ces / n, "train_consistency": cls_ / n, "val_auc": auc,
           "val_acc": acc, "lr": float(opt.learning_rate.numpy()), "sec": time.time() - t0}
    hist.append(rec); print(rec, flush=True)
    if auc is not None and auc > best:
        best, bad = auc, 0; model.save(ckpt); print("  saved best", best)
    else:
        bad += 1
        if bad % 2 == 0:
            opt.learning_rate = max(float(opt.learning_rate.numpy()) * 0.5, 1e-7)
        if bad >= a.patience:
            print("early stop"); break
if not os.path.exists(ckpt):
    model.save(ckpt)
json.dump({"config": vars(a), "seed": 42, "init": a.init, "best_val_auc_mixed_variants": best, "history": hist},
          open(f"{OUT}/{tag}.json", "w"), indent=2)
print("DONE best val AUC (mixed variants):", best)