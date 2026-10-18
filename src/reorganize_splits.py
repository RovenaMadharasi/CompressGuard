"""
reorganize_splits.py
--------------------
Reorganizes data/, data_compressed/, frames_compressed/, and frames/
into pairwise-clustered splits with zero identity/scene cross-contamination.

Each cluster {A, B} contains:
  - Real: A.mp4, B.mp4
  - Fake: A_B.mp4, B_A.mp4

Total 100 clusters:
  - Train: 70 clusters (140 real, 140 fake)
  - Val:   15 clusters (30 real, 30 fake)
  - Test:  15 clusters (30 real, 30 fake)
"""

import os
import shutil
import random
import argparse

PROJECT_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake"
SEED = 42

SPLIT_COUNTS = {
    "train": 70,
    "val": 15,
    "test": 15
}


def build_pair_mapping():
    # Discover all 200 fake videos from data_compressed/c23
    c23_dir = os.path.join(PROJECT_DIR, "data_compressed", "c23")
    fake_files = []
    for s in ["train", "val", "test"]:
        d = os.path.join(c23_dir, s, "fake")
        if os.path.isdir(d):
            fake_files.extend([f for f in os.listdir(d) if f.endswith(".mp4")])

    pair_set = set()
    for f in fake_files:
        base = os.path.splitext(f)[0]
        t, s = base.split("_")
        pair_set.add(tuple(sorted([t, s])))

    pairs = sorted(list(pair_set))
    if len(pairs) != 100:
        raise ValueError(f"Expected 100 mutual pairs, found {len(pairs)}")

    # Shuffle pairs deterministically
    rng = random.Random(SEED)
    rng.shuffle(pairs)

    # Allocate pairs to splits
    assignments = {}
    start = 0
    for split_name, count in SPLIT_COUNTS.items():
        assignments[split_name] = pairs[start:start + count]
        start += count

    # Build target split lookup for every video name
    video_to_split = {}
    for split_name, assigned_pairs in assignments.items():
        for id1, id2 in assigned_pairs:
            # Real videos
            video_to_split[f"{id1}.mp4"] = (split_name, "real")
            video_to_split[f"{id2}.mp4"] = (split_name, "real")
            # Fake videos
            video_to_split[f"{id1}_{id2}.mp4"] = (split_name, "fake")
            video_to_split[f"{id2}_{id1}.mp4"] = (split_name, "fake")

    return assignments, video_to_split


def reorganize_video_directory(base_dir, video_to_split, dry_run=True):
    """Reorganize video files: <base_dir>/<split>/<cls>/<video>.mp4"""
    print(f"\nProcessing video directory: {base_dir}")
    moves = []
    # Collect all existing videos across splits
    for s in ["train", "val", "test"]:
        for c in ["real", "fake"]:
            d = os.path.join(base_dir, s, c)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".mp4"):
                    src = os.path.join(d, f)
                    target_split, target_cls = video_to_split.get(f, (None, None))
                    if target_split is None or target_cls != c:
                        raise ValueError(f"Unrecognized or misclassified video: {f}")
                    dst = os.path.join(base_dir, target_split, target_cls, f)
                    if src != dst:
                        moves.append((src, dst))

    print(f"  Files requiring move: {len(moves)}")
    if not dry_run:
        # Move to temporary location first to prevent name collisions
        temp_dir = os.path.join(base_dir, "_reorg_temp")
        os.makedirs(temp_dir, exist_ok=True)
        staged = []
        for src, dst in moves:
            tmp = os.path.join(temp_dir, os.path.basename(src) + "_" + str(random.randint(1000, 9999)))
            shutil.move(src, tmp)
            staged.append((tmp, dst))
        for tmp, dst in staged:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(tmp, dst)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("  Moves completed successfully.")


def reorganize_frame_directory(base_dir, video_to_split, dry_run=True):
    """Reorganize frame subdirectories: <base_dir>/<split>/<cls>/<video_name>/"""
    print(f"\nProcessing frame directory: {base_dir}")
    moves = []
    for s in ["train", "val", "test"]:
        for c in ["real", "fake"]:
            d = os.path.join(base_dir, s, c)
            if not os.path.isdir(d):
                continue
            for sub in os.listdir(d):
                sub_path = os.path.join(d, sub)
                if os.path.isdir(sub_path):
                    v_file = f"{sub}.mp4"
                    target_split, target_cls = video_to_split.get(v_file, (None, None))
                    if target_split is None or target_cls != c:
                        raise ValueError(f"Unrecognized or misclassified frame folder: {sub}")
                    dst = os.path.join(base_dir, target_split, target_cls, sub)
                    if sub_path != dst:
                        moves.append((sub_path, dst))

    print(f"  Folders requiring move: {len(moves)}")
    if not dry_run:
        temp_dir = os.path.join(base_dir, "_reorg_temp")
        os.makedirs(temp_dir, exist_ok=True)
        staged = []
        for src, dst in moves:
            tmp = os.path.join(temp_dir, os.path.basename(src) + "_" + str(random.randint(1000, 9999)))
            shutil.move(src, tmp)
            staged.append((tmp, dst))
        for tmp, dst in staged:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(tmp, dst)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("  Moves completed successfully.")


def verify_clean_splits(video_to_split):
    print("\n" + "=" * 60)
    print("VERIFYING CLEAN SPLIT BALANCE AND ZERO LEAKAGE")
    print("=" * 60)
    split_ids = {"train": set(), "val": set(), "test": set()}
    split_counts = {"train": {"real": 0, "fake": 0},
                    "val":   {"real": 0, "fake": 0},
                    "test":  {"real": 0, "fake": 0}}

    for v_file, (split, cls) in video_to_split.items():
        split_counts[split][cls] += 1
        name = os.path.splitext(v_file)[0]
        if "_" in name:
            t, s = name.split("_")
            split_ids[split].add(t)
            split_ids[split].add(s)
        else:
            split_ids[split].add(name)

    for s in ["train", "val", "test"]:
        print(f"  {s.upper():5s}: {split_counts[s]['real']} real, {split_counts[s]['fake']} fake | {len(split_ids[s])} unique IDs")

    print("\nOverlap checks:")
    print("  Train & Val  overlap:", len(split_ids["train"] & split_ids["val"]))
    print("  Train & Test overlap:", len(split_ids["train"] & split_ids["test"]))
    print("  Val & Test   overlap:", len(split_ids["val"] & split_ids["test"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute reorganization moves")
    args = parser.parse_args()

    assignments, video_to_split = build_pair_mapping()
    verify_clean_splits(video_to_split)

    dry_run = not args.execute
    print(f"\nMode: {'DRY RUN (no files moved)' if dry_run else 'EXECUTE (reorganizing directories)'}")

    # 1. data
    reorganize_video_directory(os.path.join(PROJECT_DIR, "data"), video_to_split, dry_run=dry_run)

    # 2. data_compressed (c23, c40, cascade)
    for v in ["c23", "c40", "cascade"]:
        reorganize_video_directory(os.path.join(PROJECT_DIR, "data_compressed", v), video_to_split, dry_run=dry_run)

    # 3. frames_compressed (c23, c40, cascade)
    for v in ["c23", "c40", "cascade"]:
        reorganize_frame_directory(os.path.join(PROJECT_DIR, "frames_compressed", v), video_to_split, dry_run=dry_run)

    # 4. frames
    reorganize_frame_directory(os.path.join(PROJECT_DIR, "frames"), video_to_split, dry_run=dry_run)

    if dry_run:
        print("\nDry run completed. To execute moves, run: python src/reorganize_splits.py --execute")
    else:
        print("\nReorganization completed successfully! All directories are now cleanly partitioned.")


if __name__ == "__main__":
    main()
