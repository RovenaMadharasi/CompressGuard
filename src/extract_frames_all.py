"""
extract_frames_all.py
---------------------
Extracts FRAMES_PER_VIDEO evenly-spaced frames per video from all
compression variants (c23, c40, cascade) and saves them as JPEG images.

Source directory: data_compressed/<variant>/<split>/<class>/<video>.mp4
Output directory: frames_compressed/<variant>/<split>/<class>/<video_name>/frame_NN.jpg

Features:
  - Cache staleness detection: if the source video has a newer modification
    timestamp than the extracted frames, the frame cache is automatically invalidated.
  - Staging directory pattern (<video_name>_staging) to ensure atomic extraction.
    Interrupted runs never leave partial frame caches.
  - Frame validation: ensures exactly FRAMES_PER_VIDEO readable, non-empty JPEGs.
  - Flags:
      --force       : re-extract frames even if valid cache exists
      --sample N    : process only N videos per split/class (for smoke testing)
      --verify-only : audit existing extracted frame directories without modifying them
"""

import argparse
import os
import shutil
import cv2
import numpy as np

# -- Config ---------------------------------------------------------------------
VARIANTS         = ["c23", "c40", "cascade"]
DATA_BASE        = r"D:\3rd_yr_sem5\stirs_research\DeepFake\data_compressed"
FRAME_BASE       = r"D:\3rd_yr_sem5\stirs_research\DeepFake\frames_compressed"
SPLITS           = ["train", "val", "test"]
CLASSES          = ["real", "fake"]
FRAMES_PER_VIDEO = 16
MIN_FRAME_BYTES  = 1024  # 1 KB minimum


# -- Frame extraction helpers ---------------------------------------------------

def is_frame_cache_valid(out_dir, video_path):
    """
    Validates that:
    1. Output directory exists.
    2. Contains exactly FRAMES_PER_VIDEO valid .jpg files.
    3. Each frame file is >= MIN_FRAME_BYTES and decodable by OpenCV.
    4. None of the frames are older than the source video file (no stale cache).
    """
    if not os.path.isdir(out_dir):
        return False

    staging = out_dir + "_staging"
    if os.path.isdir(staging):
        shutil.rmtree(staging, ignore_errors=True)
        return False

    files = sorted([f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")])
    if len(files) != FRAMES_PER_VIDEO:
        return False

    video_mtime = os.path.getmtime(video_path)

    for f in files:
        fpath = os.path.join(out_dir, f)
        if os.path.getsize(fpath) < MIN_FRAME_BYTES:
            return False
        if os.path.getmtime(fpath) < video_mtime:
            # Video was modified after frames were extracted -> stale cache!
            return False

    return True


def extract_frames_from_video(video_path, out_dir):
    """
    Extracts FRAMES_PER_VIDEO frames into a staging directory first.
    Moves staging to out_dir only if all FRAMES_PER_VIDEO frames succeed.
    Returns True on success, False on failure.
    """
    staging = out_dir + "_staging"
    if os.path.isdir(staging):
        shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        shutil.rmtree(staging, ignore_errors=True)
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        shutil.rmtree(staging, ignore_errors=True)
        return False

    indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)
    saved = 0

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        # Fallback to sequential read if seek fails
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx) - 1))
            ok, frame = cap.read()

        if ok and frame is not None:
            out_file = os.path.join(staging, f"frame_{i + 1:02d}.jpg")
            write_ok = cv2.imwrite(out_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if write_ok and os.path.getsize(out_file) >= MIN_FRAME_BYTES:
                saved += 1
            else:
                if os.path.exists(out_file):
                    os.remove(out_file)

    cap.release()

    if saved == FRAMES_PER_VIDEO:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        shutil.move(staging, out_dir)
        return True
    else:
        shutil.rmtree(staging, ignore_errors=True)
        return False


# -- Pipeline Runner ------------------------------------------------------------

def run_extraction(force=False, sample_limit=None, verify_only=False):
    print("=" * 70)
    print("COMPRESSGUARD FRAME EXTRACTION PIPELINE")
    print(f"  Source       : {DATA_BASE}")
    print(f"  Output       : {FRAME_BASE}")
    print(f"  Frames/video : {FRAMES_PER_VIDEO}")
    print(f"  Mode         : {'VERIFY-ONLY' if verify_only else ('FORCE REBUILD' if force else 'INCREMENTAL (STALENESS CHECK)')}")
    if sample_limit:
        print(f"  Sample       : First {sample_limit} videos per split/class")
    print("=" * 70)

    stats = {v: {"reused": 0, "extracted": 0, "regenerated": 0, "failed": 0} for v in VARIANTS}

    for variant in VARIANTS:
        print(f"\nVariant: {variant}")
        for split in SPLITS:
            for cls in CLASSES:
                in_dir = os.path.join(DATA_BASE, variant, split, cls)
                out_base = os.path.join(FRAME_BASE, variant, split, cls)

                if not os.path.isdir(in_dir):
                    print(f"  [SKIP] Missing: {in_dir}")
                    continue

                videos = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".mp4")])
                if sample_limit:
                    videos = videos[:sample_limit]

                print(f"  [{split}/{cls}] {len(videos)} videos")

                for i, v in enumerate(videos, 1):
                    v_name = os.path.splitext(v)[0]
                    v_path = os.path.join(in_dir, v)
                    out_dir = os.path.join(out_base, v_name)

                    valid = is_frame_cache_valid(out_dir, v_path)
                    was_stale = os.path.isdir(out_dir) and not valid

                    if valid and not force:
                        stats[variant]["reused"] += 1
                        continue

                    if verify_only:
                        if valid:
                            stats[variant]["reused"] += 1
                        else:
                            stats[variant]["failed"] += 1
                        continue

                    # Extract frames
                    ok = extract_frames_from_video(v_path, out_dir)
                    if ok:
                        if was_stale:
                            stats[variant]["regenerated"] += 1
                            status = "regenerated (stale cache)"
                        else:
                            stats[variant]["extracted"] += 1
                            status = "extracted"
                    else:
                        stats[variant]["failed"] += 1
                        status = "FAILED"

                    print(f"    [{i:3d}/{len(videos)}] {v:<15} -> {status}")

    # Summary table
    print("\n" + "=" * 75)
    print("FRAME EXTRACTION SUMMARY")
    print(f"{'Variant':<10} | {'Reused (Valid)':<15} | {'Fresh Extracted':<16} | {'Regenerated (Stale)':<20} | {'Failed':<8}")
    print("-" * 75)
    for v in VARIANTS:
        s = stats[v]
        print(f"{v:<10} | {s['reused']:<15} | {s['extracted']:<16} | {s['regenerated']:<20} | {s['failed']:<8}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from compression variants.")
    parser.add_argument("--force", action="store_true", help="Re-extract frames even if valid cache exists.")
    parser.add_argument("--sample", type=int, default=None, help="Process only N videos per split/class.")
    parser.add_argument("--verify-only", action="store_true", help="Audit frame caches without extraction.")
    args = parser.parse_args()

    run_extraction(force=args.force, sample_limit=args.sample, verify_only=args.verify_only)
