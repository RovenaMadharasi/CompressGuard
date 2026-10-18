"""
plot_results.py
---------------
Generates visual evaluation charts for the CompressGuard research report:
  1. auc_comparison.png       : Bar chart comparing Baseline vs CompressGuard
                                with 95% video-level bootstrap confidence intervals.
  2. accuracy_comparison.png  : Side-by-side test accuracy comparison.
  3. f1_comparison.png        : Side-by-side test F1-score comparison.
  4. training_history.png     : Training and validation loss/AUC trajectories.
  5. confusion_matrices/      : Confusion matrix heatmaps for each compression variant.

Usage:
  python src/plot_results.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # Headless backend safe on Windows/servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# -- Paths -----------------------------------------------------------------------
RESULTS_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\results_compressguard"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

VARIANTS      = ["c23", "c40", "cascade"]
VARIANT_NAMES = ["CRF-23\n(Medium)", "CRF-40\n(Heavy)", "Cascade\n(2-Gen)"]

COLORS = {
    "baseline":      "#4C72B0",  # Professional steel blue
    "compressguard": "#DD8452",  # Muted warm terracotta
}


# -- Loading & Validation Helpers -----------------------------------------------

def load_eval_json(tag):
    path = os.path.join(RESULTS_DIR, f"eval_{tag}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[WARN] Error reading {path}: {e}")
        return None


def get_ci_error_bars(eval_data, variants):
    """Calculates asymmetric error bar offsets for matplotlib bar plots."""
    err_low, err_high = [], []
    for v in variants:
        entry = eval_data.get(v, {})
        auc = entry.get("auc", 0.5)
        lo  = entry.get("auc_95ci_low", auc)
        hi  = entry.get("auc_95ci_high", auc)

        # In case CI is missing or NaN
        if lo is None or np.isnan(lo):
            lo = auc
        if hi is None or np.isnan(hi):
            hi = auc

        err_low.append(max(0.0, auc - lo))
        err_high.append(max(0.0, hi - auc))

    return np.array([err_low, err_high])


# -- Plotting Functions ---------------------------------------------------------

def plot_auc_comparison(baseline, cg, common_variants):
    names = [VARIANT_NAMES[VARIANTS.index(v)] for v in common_variants]
    x = np.arange(len(common_variants))
    width = 0.35

    bl_auc = [baseline[v]["auc"] for v in common_variants]
    cg_auc = [cg[v]["auc"] for v in common_variants]

    bl_err = get_ci_error_bars(baseline, common_variants)
    cg_err = get_ci_error_bars(cg, common_variants)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2, bl_auc, width,
        yerr=bl_err, capsize=6, error_kw={"linewidth": 1.5},
        label="Baseline (Trained on c23)", color=COLORS["baseline"], alpha=0.9
    )
    bars2 = ax.bar(
        x + width / 2, cg_auc, width,
        yerr=cg_err, capsize=6, error_kw={"linewidth": 1.5},
        label="CompressGuard (Multi-Generation Robust)", color=COLORS["compressguard"], alpha=0.9
    )

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0.40, 1.05)
    ax.set_ylabel("ROC-AUC Score", fontsize=12)
    ax.set_xlabel("Compression Degradation", fontsize=12)
    ax.set_title("DeepFake Detection ROC-AUC vs Video Compression\n(With 95% Video-Level Bootstrap Confidence Intervals)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance Level (0.50)")
    ax.legend(fontsize=10, loc="lower left")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "auc_comparison.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  [SAVED] {out}")


def plot_metric_comparison(baseline, cg, common_variants, metric_key, title, out_name):
    names = [VARIANT_NAMES[VARIANTS.index(v)] for v in common_variants]
    x = np.arange(len(common_variants))
    width = 0.35

    bl_vals = [baseline[v].get(metric_key, 0.0) for v in common_variants]
    cg_vals = [cg[v].get(metric_key, 0.0) for v in common_variants]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, bl_vals, width, label="Baseline", color=COLORS["baseline"], alpha=0.9)
    bars2 = ax.bar(x + width / 2, cg_vals, width, label="CompressGuard", color=COLORS["compressguard"], alpha=0.9)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric_key.capitalize(), fontsize=12)
    ax.set_xlabel("Compression Degradation", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  [SAVED] {out}")


def plot_training_history():
    path = os.path.join(RESULTS_DIR, "training_history.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found.")
        return

    with open(path) as f:
        h = json.load(f)

    if "loss" not in h or not h["loss"]:
        print("  [SKIP] training_history.json is empty.")
        return

    epochs = range(1, len(h["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, h["loss"], "b-o", markersize=4, label="Train Loss")
    if "val_loss" in h and h["val_loss"]:
        ax1.plot(epochs, h["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax1.set_title("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if "auc" in h and h["auc"]:
        ax2.plot(epochs, h["auc"], "b-o", markersize=4, label="Train AUC")
    if "val_auc" in h and h["val_auc"]:
        ax2.plot(epochs, h["val_auc"], "r-o", markersize=4, label="Val AUC")
    ax2.set_title("ROC-AUC Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("CompressGuard Model Convergence History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "training_history.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  [SAVED] {out}")


def plot_confusion_matrices(baseline, cg, common_variants):
    cm_dir = os.path.join(PLOTS_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    for v in common_variants:
        label = VARIANT_NAMES[VARIANTS.index(v)].replace("\n", " ")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Confusion Matrices: {label}", fontsize=13, fontweight="bold")

        for ax, model_name, data, col in [
            (axes[0], "Baseline", baseline, COLORS["baseline"]),
            (axes[1], "CompressGuard", cg, COLORS["compressguard"])
        ]:
            cm = np.array(data[v]["confusion_matrix"])
            disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(model_name, fontsize=11, fontweight="bold", color=col)

        fig.tight_layout()
        out = os.path.join(cm_dir, f"confusion_matrix_{v}.png")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"  [SAVED] {out}")


# -- Main -----------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    baseline = load_eval_json("baseline")
    cg = load_eval_json("compressguard")

    print("=" * 65)
    print("COMPRESSGUARD RESULTS PLOTTING")
    print(f"  Results directory : {RESULTS_DIR}")
    print(f"  Plots directory   : {PLOTS_DIR}")
    print("=" * 65)

    if baseline and cg:
        common = [v for v in VARIANTS if v in baseline and v in cg]
        if common:
            print(f"\nGenerating comparative charts for variants: {', '.join(common)}")
            plot_auc_comparison(baseline, cg, common)
            plot_metric_comparison(baseline, cg, common, "accuracy",
                                   "DeepFake Detection Accuracy vs Compression", "accuracy_comparison.png")
            plot_metric_comparison(baseline, cg, common, "f1_score",
                                   "DeepFake Detection F1-Score vs Compression", "f1_comparison.png")
            plot_confusion_matrices(baseline, cg, common)
        else:
            print("ERROR: No common compression variants between baseline and compressguard evaluations.")
    else:
        if not baseline:
            print("\n[INFO] eval_baseline.json not found. Run:")
            print("  python src/evaluate_compressions.py --model results/xception_baseline.keras --tag baseline")
        if not cg:
            print("\n[INFO] eval_compressguard.json not found. Run:")
            print("  python src/evaluate_compressions.py --model results_compressguard/compressguard_best.keras --tag compressguard")

    print("\nGenerating training history plot...")
    plot_training_history()
    print("\nPlotting complete.\n")


if __name__ == "__main__":
    main()
