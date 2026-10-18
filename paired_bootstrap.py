import csv, glob, os
import numpy as np
from sklearn.metrics import roc_auc_score

D = "outputs/v2/eval"
MODELS = ["2b_c23", "mixed", "mixed_cons"]
VARIANTS = ["c23", "c40", "cascade"]
PAIRS = [("mixed_cons", "2b_c23"), ("mixed_cons", "mixed"), ("mixed", "2b_c23")]
N_BOOT = 5000

def load(name, variant):
    # AUC does not depend on the threshold, so any file for this model/variant works
    fs = glob.glob(f"{D}/{name}_{variant}_video_probs.csv") + glob.glob(f"{D}/{name}_{variant}_thr*_video_probs.csv")
    if not fs:
        return None
    a = np.loadtxt(sorted(fs)[0], delimiter=",", skiprows=1)
    return a[:, 0].astype(int), a[:, 1]

rng = np.random.default_rng(42)
rows = []
print(f"{'variant':8} {'A':11} {'B':11} {'AUC A':>7} {'AUC B':>7} {'diff':>7}  95% CI of diff        P(diff<=0)")
for v in VARIANTS:
    for A, B in PAIRS:
        la, lb = load(A, v), load(B, v)
        if la is None or lb is None:
            print(f"{v:8} {A:11} {B:11} missing file, skipped"); continue
        y, pa = la; y2, pb = lb
        assert len(y) == len(y2) and (y == y2).all(), "video order/labels differ"
        diffs = []
        n = len(y)
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)          # SAME videos for both models (paired)
            if len(set(y[i])) < 2:
                continue
            diffs.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
        diffs = np.array(diffs)
        a_auc, b_auc = roc_auc_score(y, pa), roc_auc_score(y, pb)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = float((diffs <= 0).mean())
        print(f"{v:8} {A:11} {B:11} {a_auc:7.3f} {b_auc:7.3f} {a_auc-b_auc:+7.3f}  [{lo:+.3f}, {hi:+.3f}]   {p:.3f}")
        rows.append([v, A, B, a_auc, b_auc, a_auc - b_auc, lo, hi, p])

with open(f"{D}/paired_bootstrap.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "A", "B", "auc_A", "auc_B", "diff", "ci_lo", "ci_hi", "p_diff_le_0"])
    w.writerows(rows)
print("Saved", f"{D}/paired_bootstrap.csv")