import os
import shutil
import random

DATASET_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\dataset"
OUTPUT_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\data"

REAL_DIR = os.path.join(DATASET_DIR, "original_sequences", "youtube", "c23", "videos")
FAKE_DIR = os.path.join(DATASET_DIR, "manipulated_sequences", "Deepfakes", "c23", "videos")

random.seed(42)

SPLITS = {
    "train": 70,  # 70 pairs = 140 real, 140 fake
    "val": 15,    # 15 pairs = 30 real, 30 fake
    "test": 15    # 15 pairs = 30 real, 30 fake
}


def prepare_paired_dataset():
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.lower().endswith(".mp4")]
    pair_set = set()
    for f in fake_files:
        base = os.path.splitext(f)[0]
        t, s = base.split("_")
        pair_set.add(tuple(sorted([t, s])))

    pairs = sorted(list(pair_set))
    if len(pairs) != sum(SPLITS.values()):
        raise ValueError(f"Expected {sum(SPLITS.values())} pairs, found {len(pairs)}")

    random.Random(42).shuffle(pairs)

    start = 0
    for split_name, count in SPLITS.items():
        assigned_pairs = pairs[start:start + count]
        start += count

        real_dst = os.path.join(OUTPUT_DIR, split_name, "real")
        fake_dst = os.path.join(OUTPUT_DIR, split_name, "fake")
        os.makedirs(real_dst, exist_ok=True)
        os.makedirs(fake_dst, exist_ok=True)

        for id1, id2 in assigned_pairs:
            # Copy real source videos
            shutil.copy2(os.path.join(REAL_DIR, f"{id1}.mp4"), os.path.join(real_dst, f"{id1}.mp4"))
            shutil.copy2(os.path.join(REAL_DIR, f"{id2}.mp4"), os.path.join(real_dst, f"{id2}.mp4"))
            # Copy fake counterpart videos
            shutil.copy2(os.path.join(FAKE_DIR, f"{id1}_{id2}.mp4"), os.path.join(fake_dst, f"{id1}_{id2}.mp4"))
            shutil.copy2(os.path.join(FAKE_DIR, f"{id2}_{id1}.mp4"), os.path.join(fake_dst, f"{id2}_{id1}.mp4"))

        print(f"{split_name}: {len(assigned_pairs) * 2} real, {len(assigned_pairs) * 2} fake (zero cross-split leakage)")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    prepare_paired_dataset()
    print("\nDataset preparation completed with pair clustering.")
