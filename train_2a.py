"""
train_2a.py
-----------
Experiment 2a: Baseline DeepFake detection on c23 face crops.
Setup:
  - Data: frames_faces/c23/{train,val}/{real,fake}
  - Architecture: Pretrained Xception backbone (weights='imagenet', frozen, training=False)
  - Classification head: Dense(128, relu, L2 1e-4) + Dropout(0.5) + Dense(1, sigmoid)
  - Preprocessing: tf.keras.applications.xception.preprocess_input inside graph (maps [0, 255] to [-1, 1])
  - Augmentation: RandomFlip, RandomRotation(0.03), RandomZoom(0.08), RandomContrast(0.10), RandomBrightness(0.08)
  - Optimizer: Adam(learning_rate=1e-4)
  - Loss: binary_crossentropy
  - Batch size: 16, Epochs: 15 (with EarlyStopping patience=3, ReduceLROnPlateau patience=2)
  - Evaluates on VAL set only using best checkpoint.
"""

import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, Model, regularizers
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

# -- Hyperparameters & Paths copied exactly from src/train.py & src/model.py --
PROJECT_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake"
DATA_BASE = os.path.join(PROJECT_DIR, "frames_faces", "c23")
OUTPUT_V2_DIR = os.path.join(PROJECT_DIR, "outputs", "v2")

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Fix seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_baseline_model():
    """Builds the exact model from src/model.py."""
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(299, 299, 3)
    )
    base_model.trainable = False

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.10),
        layers.RandomBrightness(0.08)
    ], name="augmentation")

    inputs = layers.Input(shape=(299, 299, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.xception.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="xception_baseline_head_only")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


def collect_split_paths(split):
    paths = []
    labels = []
    for class_name, label in [("real", 0), ("fake", 1)]:
        class_dir = os.path.join(DATA_BASE, split, class_name)
        files = sorted(glob.glob(os.path.join(class_dir, "*", "*.jpg")))
        paths.extend(files)
        labels.extend([label] * len(files))

    combined = list(zip(paths, labels))
    random.Random(SEED).shuffle(combined)
    paths, labels = zip(*combined)
    return list(paths), list(labels)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 2048),
            seed=SEED,
            reshuffle_each_iteration=True
        )
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def evaluate_val_set(model, val_paths, val_labels):
    """Evaluates the validation set frame-by-frame using best checkpoint."""
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    probs = model.predict(val_ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    labels = np.array(val_labels, dtype=int)

    acc = float(accuracy_score(labels, preds))
    auc = float(roc_auc_score(labels, probs))
    cm = confusion_matrix(labels, preds)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    recall_real = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    recall_fake = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec_real = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    prec_fake = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1_real = float(2 * prec_real * recall_real / (prec_real + recall_real)) if (prec_real + recall_real) > 0 else 0.0
    f1_fake = float(2 * prec_fake * recall_fake / (prec_fake + recall_fake)) if (prec_fake + recall_fake) > 0 else 0.0

    return {
        "val_auc": round(auc, 6),
        "val_accuracy": round(acc, 6),
        "recall_real": round(recall_real, 6),
        "recall_fake": round(recall_fake, 6),
        "precision_real": round(prec_real, 6),
        "precision_fake": round(prec_fake, 6),
        "f1_real": round(f1_real, 6),
        "f1_fake": round(f1_fake, 6),
        "confusion_matrix": {
            "TN_real": tn,
            "FP_real_pred_fake": fp,
            "FN_fake_pred_real": fn,
            "TP_fake": tp
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2a: Baseline on c23 face crops")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch on tiny subset to verify pipeline")
    args = parser.parse_args()

    os.makedirs(OUTPUT_V2_DIR, exist_ok=True)

    print("=" * 76)
    print("EXPERIMENT 2a: BASELINE ON c23 FACE CROPS (FROZEN BACKBONE, HEAD ONLY)")
    print("=" * 76)

    # 1. Print data path being loaded
    print(f"Data Path Being Loaded : {DATA_BASE}")
    print(f"  Train Real Directory : {os.path.join(DATA_BASE, 'train', 'real')}")
    print(f"  Train Fake Directory : {os.path.join(DATA_BASE, 'train', 'fake')}")
    print(f"  Val Real Directory   : {os.path.join(DATA_BASE, 'val', 'real')}")
    print(f"  Val Fake Directory   : {os.path.join(DATA_BASE, 'val', 'fake')}")

    # 2. Collect and count frames per split and class
    train_paths, train_labels = collect_split_paths("train")
    val_paths, val_labels = collect_split_paths("val")

    n_train_real = train_labels.count(0)
    n_train_fake = train_labels.count(1)
    n_val_real = val_labels.count(0)
    n_val_fake = val_labels.count(1)

    print("\nImage Counts per Split and Class:")
    print(f"  TRAIN : Total = {len(train_paths):5d} | Real = {n_train_real:5d} | Fake = {n_train_fake:5d}")
    print(f"  VAL   : Total = {len(val_paths):5d} | Real = {n_val_real:5d} | Fake = {n_val_fake:5d}")

    # 3. Label mapping
    print("\nLabel Mapping:")
    print("  Class 'real' -> 0")
    print("  Class 'fake' -> 1")

    # 4. Check one batch value range after preprocessing
    sample_img, _ = load_image(train_paths[0], train_labels[0])
    raw_min = float(tf.reduce_min(sample_img))
    raw_max = float(tf.reduce_max(sample_img))
    preprocessed_img = tf.keras.applications.xception.preprocess_input(sample_img)
    prep_min = float(tf.reduce_min(preprocessed_img))
    prep_max = float(tf.reduce_max(preprocessed_img))

    print("\nBatch Value Range Inspection:")
    print(f"  Raw Decoded Image Range         : [{raw_min:.2f}, {raw_max:.2f}] (expected [0.0, 255.0])")
    print(f"  After Xception Preprocessing    : [{prep_min:.4f}, {prep_max:.4f}] (expected [-1.0, 1.0])")

    # 5. Build model and inspect trainable parameters
    model = build_baseline_model()
    trainable_count = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable_count = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count

    print("\nModel Parameter Counts:")
    print(f"  Total Parameters     : {total_count:,}")
    print(f"  Trainable Parameters : {trainable_count:,} (Head only: Dense 128 + Dense 1)")
    print(f"  Non-Trainable Params : {non_trainable_count:,} (Frozen Xception backbone)")

    # Smoke test mode adjustment
    epochs_to_run = EPOCHS
    if args.smoke_test:
        print("\n[SMOKE TEST MODE ACTIVATED] Running 1 epoch on 64 train and 32 val frames...")
        train_paths = train_paths[:64]
        train_labels = train_labels[:64]
        val_paths = val_paths[:32]
        val_labels = val_labels[:32]
        epochs_to_run = 1

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, shuffle=False)

    ckpt_path = os.path.join(OUTPUT_V2_DIR, "2a_c23_head_only_best.keras")
    final_path = os.path.join(OUTPUT_V2_DIR, "2a_c23_head_only_final.keras")
    json_path = os.path.join(OUTPUT_V2_DIR, "2a_c23_head_only.json")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n" + "-" * 76)
    print(f"STARTING TRAINING ({epochs_to_run} max epochs, batch_size={BATCH_SIZE}, lr=1e-4)...")
    print("-" * 76)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_to_run,
        callbacks=callbacks
    )

    # Save final model
    model.save(final_path)

    # Evaluate using the best saved checkpoint on the VAL set
    print("\n" + "=" * 76)
    print("EVALUATING BEST CHECKPOINT ON VALIDATION SET (VAL SET ONLY)")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 76)

    best_model = tf.keras.models.load_model(ckpt_path)
    val_metrics = evaluate_val_set(best_model, val_paths, val_labels)

    print(f"\nValidation Results (Best Checkpoint):")
    print(f"  Val ROC-AUC     : {val_metrics['val_auc']:.4f}")
    print(f"  Val Accuracy    : {val_metrics['val_accuracy']:.4f}")
    print(f"  Recall (Real)   : {val_metrics['recall_real']:.4f} (TN={val_metrics['confusion_matrix']['TN_real']})")
    print(f"  Recall (Fake)   : {val_metrics['recall_fake']:.4f} (TP={val_metrics['confusion_matrix']['TP_fake']})")
    print(f"  Precision (Real): {val_metrics['precision_real']:.4f}")
    print(f"  Precision (Fake): {val_metrics['precision_fake']:.4f}")
    print(f"  F1 (Real)       : {val_metrics['f1_real']:.4f}")
    print(f"  F1 (Fake)       : {val_metrics['f1_fake']:.4f}")
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"               Pred Real   Pred Fake")
    print(f"  Actual Real    {val_metrics['confusion_matrix']['TN_real']:5d}       {val_metrics['confusion_matrix']['FP_real_pred_fake']:5d}")
    print(f"  Actual Fake    {val_metrics['confusion_matrix']['FN_fake_pred_real']:5d}       {val_metrics['confusion_matrix']['TP_fake']:5d}")

    # Build history record
    history_records = []
    num_epochs_trained = len(history.history["loss"])
    for ep in range(num_epochs_trained):
        history_records.append({
            "epoch": ep + 1,
            "train_loss": round(float(history.history["loss"][ep]), 6),
            "train_accuracy": round(float(history.history["accuracy"][ep]), 6),
            "train_auc": round(float(history.history["auc"][ep]), 6),
            "val_loss": round(float(history.history["val_loss"][ep]), 6),
            "val_accuracy": round(float(history.history["val_accuracy"][ep]), 6),
            "val_auc": round(float(history.history["val_auc"][ep]), 6),
            "lr": float(history.history.get("learning_rate", [1e-4])[ep]) if "learning_rate" in history.history else 1e-4
        })

    config_log = {
        "experiment": "2a_c23_head_only",
        "data_path": DATA_BASE,
        "seed": SEED,
        "image_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "max_epochs": epochs_to_run,
        "epochs_trained": num_epochs_trained,
        "learning_rate_initial": 1e-4,
        "optimizer": "Adam",
        "loss": "binary_crossentropy",
        "decision_threshold": 0.5,
        "augmentation": ["RandomFlip(horizontal)", "RandomRotation(0.03)", "RandomZoom(0.08)", "RandomContrast(0.10)", "RandomBrightness(0.08)"],
        "backbone": "Xception (ImageNet pretrained, frozen)",
        "trainable_parameters": trainable_count,
        "non_trainable_parameters": non_trainable_count,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "best_checkpoint": ckpt_path,
        "final_checkpoint": final_path,
        "validation_evaluation_best_checkpoint": val_metrics,
        "epoch_history": history_records
    }

    with open(json_path, "w") as f:
        json.dump(config_log, f, indent=2)

    print("\n" + "=" * 76)
    print(f"Results and configuration logged to: {json_path}")
    print("EXPERIMENT 2a COMPLETED SUCCESSFULLY.")
    print("=" * 76)


if __name__ == "__main__":
    main()
