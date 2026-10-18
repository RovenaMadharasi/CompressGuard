"""
train_compressguard.py
----------------------
Trains the CompressGuard deepfake detector on a balanced mix of compression
variants (c23, c40, cascade).

Architectural & Training Strategy:
----------------------------------
1. Architecture:
   - Uses Xception backbone with ImageNet pretrained weights.
   - Internal Rescaling layer maps [0, 255] RGB input to [-1, 1].
   - Mild data augmentations (flip, slight rotation, zoom, brightness/contrast)
     prevent spatial overfitting while preserving compression cues.
   - GlobalAveragePooling2D reduces the 2048-dim feature maps.
   - Dense classification head (128 units, ReLU, L2 regularization) + Dropout(0.5)
   - Final sigmoid output for Real (0) vs Fake (1).

2. Two-Stage Training to Prevent Gradient Shock:
   - Stage 1 (Warmup):
     Backbone is completely frozen (trainable=False).
     Trains only the dense head with Adam(lr=1e-4) until convergence.
     This ensures the randomly initialized head learns meaningful weights
     before any backbone layers are touched.
   - Stage 2 (Fine-Tuning - Optional/Configurable):
     Unfreezes only the topmost separable conv block (block14).
     All BatchNormalization layers remain frozen to preserve running statistics.
     Trained with a small learning rate (5e-6) and ReduceLROnPlateau.

3. Reproducibility & Validation:
   - Validates that train and val splits exist, are non-empty, and are balanced.
   - Fixes random seed (42).
   - Saves best checkpoint strictly based on validation AUC (val_auc).
   - Records full epoch-by-epoch metrics to CSV and JSON.

Usage:
  python src/train_compressguard.py --warmup-epochs 5 --finetune-epochs 3
  python src/train_compressguard.py --quick-test  # Fast 1-epoch sanity check
"""

import argparse
import glob
import json
import os
import random
import sys

import tensorflow as tf
from model import create_model

# -- Config ---------------------------------------------------------------------
FRAME_BASE  = r"D:\3rd_yr_sem5\stirs_research\DeepFake\frames_compressed"
RESULTS_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\results_compressguard"
VARIANTS    = ["c23", "c40", "cascade"]

IMG_SIZE       = (299, 299)
BATCH_SIZE     = 16
DEFAULT_WARMUP = 5
DEFAULT_FINE   = 3
SEED           = 42
AUTOTUNE       = tf.data.AUTOTUNE


# -- Data Loading & Validation --------------------------------------------------

def collect_split_frames(split, variants=VARIANTS):
    """
    Collects frame paths and labels across all specified compression variants.
    Ensures 0 = Real, 1 = Fake.
    """
    paths, labels = [], []
    for variant in variants:
        for cls, lbl in [("real", 0), ("fake", 1)]:
            pattern = os.path.join(FRAME_BASE, variant, split, cls, "*", "*.jpg")
            files = sorted(glob.glob(pattern))
            paths.extend(files)
            labels.extend([lbl] * len(files))

    if not paths:
        raise RuntimeError(f"No frames found for {split} split under {FRAME_BASE}.")

    # Deterministic shuffle
    combined = list(zip(paths, labels))
    random.Random(SEED).shuffle(combined)
    paths, labels = zip(*combined)
    return list(paths), list(labels)


def validate_split_balance(name, paths, labels):
    """Verifies that split has adequate samples and is balanced."""
    n_real = labels.count(0)
    n_fake = labels.count(1)
    total = len(labels)
    if total == 0:
        raise ValueError(f"{name} split is empty!")

    balance_ratio = min(n_real, n_fake) / max(n_real, n_fake)
    print(f"  {name:<6}: {total:6d} frames (Real: {n_real}, Fake: {n_fake}, Balance Ratio: {balance_ratio:.3f})")
    if balance_ratio < 0.7:
        print(f"  [WARNING] Severe imbalance in {name} split (ratio < 0.7).")


def load_raw_image(path, label):
    """
    Decodes JPEG to [0, 255] RGB float tensor.
    Preprocessing is intentionally inside the model graph.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, tf.cast(label, tf.float32)


def make_tf_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_raw_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# -- Fine-Tuning Helper ---------------------------------------------------------

def unfreeze_top_block(model, lr=5e-6):
    """
    Unfreezes only block14 (the final separable conv block) of Xception.
    All BatchNormalization layers remain frozen to prevent running stat collapse.
    """
    try:
        base = model.get_layer("xception")
    except ValueError:
        print("  [WARN] 'xception' layer not found. Continuing with frozen base.")
        return model

    base.trainable = True
    trainable_layers = 0

    for layer in base.layers:
        # Keep BatchNormalization frozen regardless
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        # Only unfreeze layers belonging to block14
        elif "block14" in layer.name:
            layer.trainable = True
            trainable_layers += 1
        else:
            layer.trainable = False

    print(f"  Fine-Tuning Configuration: {trainable_layers} non-BatchNorm layers in block14 unfrozen.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# -- Main Training Loop ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CompressGuard DeepFake detector.")
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP, help="Warmup epochs (default: 5)")
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINE, help="Fine-tuning epochs (default: 3)")
    parser.add_argument("--sample", type=int, default=None, help="Limit number of train/val frames for smoke testing")
    parser.add_argument("--quick-test", action="store_true", help="Run 1 warmup epoch on tiny subset to verify pipeline")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("COMPRESSGUARD TRAINING PIPELINE")
    print(f"  Variants    : {', '.join(VARIANTS)}")
    print(f"  Warmup      : {args.warmup_epochs} epochs (Head only, lr=1e-4)")
    print(f"  Fine-tuning : {args.finetune_epochs} epochs (Block14 only, lr=5e-6)")
    print("=" * 70)

    # GPU / CPU Configuration
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {[g.name for g in gpus]}")
    else:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        print("No GPU detected. Configured 8 CPU worker threads for stable training.")

    # Collect data
    print("\nLoading and validating frame datasets...")
    train_paths, train_labels = collect_split_frames("train")
    val_paths, val_labels = collect_split_frames("val")

    if args.quick_test:
        print("\n[QUICK-TEST MODE] Subsetting to 128 train and 64 val frames for fast dry-run.")
        train_paths, train_labels = train_paths[:128], train_labels[:128]
        val_paths, val_labels = val_paths[:64], val_labels[:64]
        args.warmup_epochs = 1
        args.finetune_epochs = 0
    elif args.sample:
        train_paths, train_labels = train_paths[:args.sample], train_labels[:args.sample]
        val_paths, val_labels = val_paths[:args.sample // 4], val_labels[:args.sample // 4]

    validate_split_balance("Train", train_paths, train_labels)
    validate_split_balance("Val", val_paths, val_labels)

    train_ds = make_tf_dataset(train_paths, train_labels, shuffle=True)
    val_ds   = make_tf_dataset(val_paths, val_labels, shuffle=False)

    # Initialize model
    print("\nInitializing Xception architecture...")
    model = create_model()

    ckpt_path = os.path.join(RESULTS_DIR, "compressguard_best.keras")
    final_path = os.path.join(RESULTS_DIR, "compressguard_final.keras")
    history_json_path = os.path.join(RESULTS_DIR, "training_history.json")

    callbacks_warmup = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(RESULTS_DIR, "training_log_warmup.csv")),
    ]

    # Stage 1: Warmup
    print(f"\n{'-'*60}")
    print(f"STAGE 1: WARMUP TRAINING ({args.warmup_epochs} epochs)")
    print(f"{'-'*60}")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.warmup_epochs,
        callbacks=callbacks_warmup,
    )

    combined_history = {k: [float(v) for v in vals] for k, vals in h1.history.items()}

    # Stage 2: Fine-Tuning (if requested)
    if args.finetune_epochs > 0:
        print(f"\n{'-'*60}")
        print(f"STAGE 2: FINE-TUNING ({args.finetune_epochs} epochs)")
        print(f"{'-'*60}")
        model = unfreeze_top_block(model, lr=5e-6)

        callbacks_finetune = [
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=2, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.CSVLogger(os.path.join(RESULTS_DIR, "training_log_finetune.csv")),
        ]

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.finetune_epochs,
            callbacks=callbacks_finetune,
        )

        for k, vals in h2.history.items():
            if k in combined_history:
                combined_history[k].extend([float(v) for v in vals])
            else:
                combined_history[k] = [float(v) for v in vals]

    # Save final model and complete history
    model.save(final_path)
    with open(history_json_path, "w") as f:
        json.dump(combined_history, f, indent=2)

    best_val_auc = max(combined_history["val_auc"]) if "val_auc" in combined_history else 0.0
    print("\n" + "=" * 70)
    print("TRAINING RUN COMPLETE")
    print(f"  Best Val AUC       : {best_val_auc:.4f}")
    print(f"  Best Model Saved   : {ckpt_path}")
    print(f"  Final Model Saved  : {final_path}")
    print(f"  History Saved      : {history_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
