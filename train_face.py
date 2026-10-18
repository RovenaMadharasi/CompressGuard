import argparse, json, os
import numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from tensorflow.keras import layers, callbacks

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["head", "finetune"], required=True)
p.add_argument("--variant", default="c23")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch", type=int, default=32)   # use 16 if you run out of memory
p.add_argument("--init", default=None)            # head checkpoint, needed for finetune
p.add_argument("--smoke", action="store_true")
a = p.parse_args()

tf.keras.utils.set_random_seed(42)
ROOT, OUT = "frames_faces", "outputs/v2"
os.makedirs(OUT, exist_ok=True)
tag = ("smoke_" if a.smoke else "") + ("2a_%s_head" if a.mode == "head" else "2b_%s_finetune") % a.variant
ckpt = f"{OUT}/{tag}.keras"

def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        f"{ROOT}/{a.variant}/{split}", labels="inferred", label_mode="binary",
        class_names=["real", "fake"],            # real=0, fake=1
        image_size=(299, 299), batch_size=a.batch, shuffle=shuffle, seed=42)

train_raw, val_raw = load("train", True), load("val", a.smoke)
val_paths = list(val_raw.file_paths)             # same order as predictions (shuffle=False)
n_train = len(train_raw.file_paths)
print("Data:", f"{ROOT}/{a.variant}/{{train,val}}", "| train imgs:", n_train, "| val imgs:", len(val_paths))
print("Class names (index = label):", train_raw.class_names)
for name, ds in [("train", train_raw), ("val", val_raw)]:
    y = np.concatenate([b[1].numpy().ravel() for b in ds])
    print(f"{name}: real={int((y == 0).sum())} fake={int((y == 1).sum())}")

train, val = train_raw, val_raw
if a.smoke:
    train, val, a.epochs = train_raw.take(3), val_raw.take(2).cache(), 1
    val_paths = val_paths[: 2 * a.batch]
train, val = train.prefetch(2), val.prefetch(2)

xb, _ = next(iter(train_raw))
print("Raw batch range:", float(tf.reduce_min(xb)), float(tf.reduce_max(xb)),
      "-> after preprocess:", float(tf.reduce_min(tf.keras.applications.xception.preprocess_input(xb))),
      float(tf.reduce_max(tf.keras.applications.xception.preprocess_input(xb))))

metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
if a.mode == "head":
    base = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(299, 299, 3))
    base.trainable = False
    inp = layers.Input((299, 299, 3))
    x = layers.RandomFlip("horizontal")(inp)
    x = tf.keras.applications.xception.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    lr = 1e-3
else:
    model = tf.keras.models.load_model(a.init)
    base = model.get_layer("xception")
    base.trainable = True
    for l in base.layers:   # unfreeze only block13/block14 convs; BatchNorm stays frozen
        l.trainable = l.name.startswith(("block13", "block14")) and not isinstance(l, layers.BatchNormalization)
    n_bn = sum(isinstance(l, layers.BatchNormalization) for l in base.layers)
    n_bn_trainable = sum(isinstance(l, layers.BatchNormalization) and l.trainable for l in base.layers)
    print(f"BatchNorm layers: {n_bn}, trainable: {n_bn_trainable} (must be 0)")
    lr = 1e-5

model.compile(tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=metrics)
print("Trainable params:", sum(int(np.prod(w.shape)) for w in model.trainable_weights))

cbs = [callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max", save_best_only=True),
       callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
       callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-7)]
hist = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=cbs)

best = tf.keras.models.load_model(ckpt)
y_true = np.concatenate([b[1].numpy().ravel() for b in val]).astype(int)
y_prob = best.predict(val).ravel()
y_pred = (y_prob >= 0.5).astype(int)
res = {"tag": tag, "mode": a.mode, "variant": a.variant, "lr": lr, "batch": a.batch, "seed": 42,
       "val_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None, "val_acc": float((y_pred == y_true).mean()),
       "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y_true, y_pred).tolist(),
       "report": classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True),
       "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()}}
json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
np.savetxt(f"{OUT}/{tag}_val_preds.csv", np.c_[y_true, y_prob], delimiter=",", header="label,prob", comments="")
print(json.dumps({k: res[k] for k in ["val_auc", "val_acc", "confusion_matrix [[TN,FP],[FN,TP]]"]}, indent=2))
print("Per-epoch val_auc:", [round(v, 4) for v in hist.history["val_auc"]])