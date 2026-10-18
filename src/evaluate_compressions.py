"""
evaluate_compressions.py
------------------------
Evaluates a trained deepfake detection model across three compression variants:
  - c23     : medium compression (CRF 23, single generation)
  - c40     : heavy compression  (CRF 40, single generation)
  - cascade : two-generation     (CRF 23 -> CRF 40)

Metrics computed per variant:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC
  - Video-level Bootstrap 95% Confidence Intervals (resampling by video cluster)
  - Confusion Matrix

Critical Bug Fix:
  In earlier versions, load_image() applied xception.preprocess_input()
  externally even though model.py already includes preprocessing inside
  the network graph. This caused DOUBLE PREPROCESSING, destroying all
  image signals into a flat ~ -1.0 constant field and generating spurious
  17% or 50% accuracy figures.
  Here, load_image() produces raw [0, 255] RGB tensors as expected by the model.

Usage:
  python src/evaluate_compressions.py --model results/xception_baseline.keras --tag baseline
  python src/evaluate_compressions.py --model results_compressguard/compressguard_best.keras --tag compressguard
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# -- Config ---------------------------------------------------------------------
FRAME_BASE  = r"D:\3rd_yr_sem5\stirs_research\DeepFake\frames_compressed"
RESULTS_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\results_compressguard"
VARIANTS    = ["c23", "c40", "cascade"]
IMG_SIZE    = (299, 299)
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE


# -- Data Collection & Loading --------------------------------------------------

def collect_test_frames(variant):
    """
    Collects test-split frame paths, binary labels (0=real, 1=fake),
    and video cluster IDs for video-level bootstrap uncertainty estimation.
    """
    paths, labels, video_ids = [], [], []
    vid_counter = 0

    for cls, lbl in [("real", 0), ("fake", 1)]:
        base = os.path.join(FRAME_BASE, variant, "test", cls)
        if not os.path.isdir(base):
            continue
        for video_dir in sorted(os.listdir(base)):
            full = os.path.join(base, video_dir)
            if not os.path.isdir(full):
                continue
            frames = sorted(glob.glob(os.path.join(full, "*.jpg")))
            if not frames:
                continue
            paths.extend(frames)
            labels.extend([lbl] * len(frames))
            video_ids.extend([vid_counter] * len(frames))
            vid_counter += 1

    return paths, np.array(labels, dtype=int), np.array(video_ids, dtype=int)


def load_raw_image(path, label):
    """
    Loads and decodes JPEG image in [0, 255] range.
    CRITICAL: Does NOT call preprocess_input here because the model graph
    handles preprocessing internally.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, tf.cast(label, tf.float32)


# -- Video-Level Bootstrap Uncertainty ------------------------------------------

def video_level_bootstrap_auc(labels, probs, video_ids, n_iterations=1000, confidence=0.95):
    """
    Computes 95% bootstrap confidence interval on AUC by resampling whole video clusters.
    Frames from the same video share identity and background; clustering by video
    prevents unrealistically narrow confidence bounds.
    """
    unique_vids = np.unique(video_ids)
    rng = np.random.default_rng(42)
    aucs = []

    for _ in range(n_iterations):
        sampled_vids = rng.choice(unique_vids, size=len(unique_vids), replace=True)
        idx = np.concatenate([np.where(video_ids == v)[0] for v in sampled_vids])
        s_labels = labels[idx]
        s_probs  = probs[idx]

        if len(np.unique(s_labels)) < 2:
            continue
        aucs.append(roc_auc_score(s_labels, s_probs))

    if len(aucs) < 50:
        return float("nan"), float("nan")

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(aucs, alpha * 100))
    hi = float(np.percentile(aucs, (1.0 - alpha) * 100))
    return lo, hi


# -- Evaluation Logic -----------------------------------------------------------

def evaluate_variant(model, variant):
    paths, labels, video_ids = collect_test_frames(variant)
    if len(paths) == 0:
        print(f"  [SKIP] No test frames found for '{variant}'")
        return None

    n_videos = len(np.unique(video_ids))
    if len(np.unique(labels)) < 2:
        print(f"  [SKIP] Only one class present in test set for '{variant}'")
        return None

    ds = (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(load_raw_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    probs = model.predict(ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    auc = float(roc_auc_score(labels, probs))
    ci_lo, ci_hi = video_level_bootstrap_auc(labels, probs, video_ids)

    cm = confusion_matrix(labels, preds)

    print(f"\n{'-'*60}")
    print(f"  VARIANT: {variant.upper()} ({len(paths)} frames from {n_videos} videos)")
    print(f"{'-'*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}  (95% Video-Level CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  Mean Fake Prob on Real: {probs[labels == 0].mean():.4f}")
    print(f"  Mean Fake Prob on Fake: {probs[labels == 1].mean():.4f}")
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"               Pred Real   Pred Fake")
    print(f"    Actual Real   {cm[0, 0]:5d}       {cm[0, 1]:5d}")
    print(f"    Actual Fake   {cm[1, 0]:5d}       {cm[1, 1]:5d}")

    return {
        "n_frames": int(len(paths)),
        "n_videos": int(n_videos),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "auc": round(auc, 6),
        "auc_95ci_low": round(ci_lo, 6) if not np.isnan(ci_lo) else None,
        "auc_95ci_high": round(ci_hi, 6) if not np.isnan(ci_hi) else None,
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake model across compression variants.")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model file")
    parser.add_argument("--tag", type=str, required=True, help="Model tag (e.g., baseline, compressguard)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    print("=" * 65)
    print(f"EVALUATING MODEL: {args.model}")
    print(f"TAG: {args.tag}")
    print("=" * 65)

    model = tf.keras.models.load_model(args.model)

    all_results = {}
    for v in VARIANTS:
        res = evaluate_variant(model, v)
        if res is not None:
            all_results[v] = res

    # Save to JSON
    out_path = os.path.join(RESULTS_DIR, f"eval_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY RESULTS: {args.tag.upper()}")
    print(f"{'Variant':<10} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'AUC':<8} | {'95% CI (Video-Level)'}")
    print("-" * 80)
    for v, r in all_results.items():
        ci_str = f"[{r['auc_95ci_low']:.4f}, {r['auc_95ci_high']:.4f}]" if r["auc_95ci_low"] is not None else "N/A"
        print(f"{v:<10} | {r['accuracy']:<8.4f} | {r['precision']:<8.4f} | {r['recall']:<8.4f} | {r['f1_score']:<8.4f} | {r['auc']:<8.4f} | {ci_str}")
    print("=" * 80)
    print(f"Results written to: {out_path}\n")


if __name__ == "__main__":
    main()
