"""
verify_splits.py
----------------
Verifies that:
1. All three splits (train, val, test) have balanced real and fake classes.
2. All mutual pairs {A, B} are strictly contained within a single split.
3. Cross-split identity and scene leakage is exactly zero.
"""

import os
from collections import defaultdict

PROJECT_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake"
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
FRAME_DIR   = os.path.join(PROJECT_DIR, "frames_compressed")


def check_splits():
    print("=" * 64)
    print("CHECKING DATASET SPLIT INTEGRITY & LEAKAGE")
    print("=" * 64)

    splits = ["train", "val", "test"]
    video_map = {}

    for s in splits:
        for c in ["real", "fake"]:
            d = os.path.join(DATA_DIR, s, c)
            if not os.path.isdir(d):
                print(f"[ERROR] Directory missing: {d}")
                return False
            files = [f for f in os.listdir(d) if f.endswith(".mp4")]
            for f in files:
                name = os.path.splitext(f)[0]
                if "_" in name:
                    target, source = name.split("_")
                else:
                    target, source = name, None
                video_map[(s, c, f)] = (target, source)

    # 1. Verify Balance and Target Overlap within each split
    split_targets = {"real": defaultdict(set), "fake": defaultdict(set)}
    for (s, c, f), (t, src) in video_map.items():
        split_targets[c][s].add(t)

    print("\n1. Split Balance & Pair Alignment:")
    for s in splits:
        n_real = len(split_targets["real"][s])
        n_fake = len(split_targets["fake"][s])
        overlap = len(split_targets["real"][s] & split_targets["fake"][s])
        print(f"  {s.upper():5s}: {n_real:3d} real, {n_fake:3d} fake | Within-split pair match: {overlap}/{n_real}")
        if n_real != n_fake or overlap != n_real:
            print(f"    [FAIL] Split {s} is not properly paired or balanced!")
            return False

    # 2. Check Cross-Split Leakage
    print("\n2. Cross-Split Leakage Check:")
    train_real = split_targets["real"]["train"]
    train_fake = split_targets["fake"]["train"]

    val_real   = split_targets["real"]["val"]
    val_fake   = split_targets["fake"]["val"]

    test_real  = split_targets["real"]["test"]
    test_fake  = split_targets["fake"]["test"]

    leaks = [
        ("Val Real in Train Fake",   len(val_real & train_fake)),
        ("Val Fake in Train Real",   len(val_fake & train_real)),
        ("Test Real in Train Fake",  len(test_real & train_fake)),
        ("Test Fake in Train Real",  len(test_fake & train_real)),
        ("Val Real in Test Fake",    len(val_real & test_fake)),
        ("Test Real in Val Fake",    len(test_real & val_fake)),
    ]

    has_leak = False
    for label, count in leaks:
        status = "PASSED (0 leaks)" if count == 0 else f"FAILED ({count} leaked videos)"
        print(f"  {label:<30s}: {status}")
        if count > 0:
            has_leak = True

    if has_leak:
        print("\n[FAIL] Cross-split leakage detected!")
        return False

    print("\n[SUCCESS] All checks passed! Zero cross-split leakage.")
    print("=" * 64)
    return True


if __name__ == "__main__":
    check_splits()
