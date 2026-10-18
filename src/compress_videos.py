"""
compress_videos.py
------------------
Simulates multi-generation social-media re-encoding using FFmpeg.

Source directory: data/ (read-only, untouched)
Output directory: data_compressed/

Compression variants produced:
  - c23     : verbatim copy of source c23 videos (baseline benchmark)
  - c40     : single-pass re-encode at CRF 40 (heavy compression)
  - cascade : two-generation re-encode (source -> CRF 23 -> CRF 40)
              accurately simulating repeated social-media upload/download cycles.

Features:
  - Preserves data/ as strictly read-only.
  - Video integrity validation: checks decodability, frame count, resolution, and mtime.
  - Detects and regenerates stale or corrupt outputs.
  - Safe staging directories with atomic replacement to prevent partial writes.
  - Command-line flags:
      --force       : force rebuild of all variants
      --sample N    : process only N videos per split/class (for smoke testing)
      --verify-only : audit existing outputs without re-encoding
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import cv2

# -- Paths ----------------------------------------------------------------------
SOURCE_BASE = r"D:\3rd_yr_sem5\stirs_research\DeepFake\data"
OUTPUT_BASE = r"D:\3rd_yr_sem5\stirs_research\DeepFake\data_compressed"
SPLITS      = ["train", "val", "test"]
CLASSES     = ["real", "fake"]
MIN_FILE_BYTES = 4096  # 4 KB minimum for valid MP4 header + payload


# -- FFmpeg & Video Helpers -----------------------------------------------------

def check_ffmpeg():
    """Verify ffmpeg is available and responsive."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print("ERROR: ffmpeg returned non-zero code.")
            sys.exit(1)
        version_line = r.stdout.splitlines()[0] if r.stdout else "unknown"
        print(f"FFmpeg detected: {version_line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: ffmpeg is not found on PATH or timed out.")
        print("Ensure FFmpeg bin folder is added to system PATH.")
        sys.exit(1)


def get_video_info(video_path):
    """
    Inspect a video using OpenCV VideoCapture.
    Returns (is_valid, frame_count, fps, width, height)
    """
    if not os.path.exists(video_path) or os.path.getsize(video_path) < MIN_FILE_BYTES:
        return False, 0, 0.0, 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, 0, 0.0, 0, 0

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    is_valid = (count > 0 and width > 0 and height > 0)
    return is_valid, count, fps, width, height


def is_output_valid(output_path, source_info, source_mtime):
    """
    Check if an existing compressed video is fully valid:
    1. File exists and size >= MIN_FILE_BYTES.
    2. Modification time is not older than source (not stale).
    3. Video is decodable, frame count is within tolerance of source, dimensions match.
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) < MIN_FILE_BYTES:
        return False

    # Check for stale cache (output older than source video)
    if os.path.getmtime(output_path) < source_mtime:
        return False

    valid, out_count, _, out_w, out_h = get_video_info(output_path)
    if not valid:
        return False

    src_count, _, src_w, src_h = source_info[1], source_info[2], source_info[3], source_info[4]
    # Frame count should match within +/- 5 frames (codec padding/B-frame differences)
    if abs(out_count - src_count) > 5:
        return False
    if out_w != src_w or out_h != src_h:
        return False

    return True


def ffmpeg_encode(input_path, output_path, crf):
    """
    Encode input to output at specified CRF using libx264.
    Encodes to a temporary staging file first, then atomically renames.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    staging_path = output_path + ".tmp.mp4"
    if os.path.exists(staging_path):
        os.remove(staging_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-an",
        staging_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(staging_path):
            os.remove(staging_path)
        return False

    if not os.path.exists(staging_path) or os.path.getsize(staging_path) < MIN_FILE_BYTES:
        if os.path.exists(staging_path):
            os.remove(staging_path)
        return False

    # Atomic move
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(staging_path, output_path)
    return True


def cascade_encode(input_path, output_path, crf1=23, crf2=40):
    """
    Two-generation encode:
      Generation 1: source -> CRF 23 intermediate
      Generation 2: intermediate -> CRF 40 output
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        gen1_path = os.path.join(tmpdir, "gen1_c23.mp4")
        ok1 = ffmpeg_encode(input_path, gen1_path, crf=crf1)
        if not ok1:
            return False
        ok2 = ffmpeg_encode(gen1_path, output_path, crf=crf2)
        return ok2


# -- Core Pipeline --------------------------------------------------------------

def process_video_variants(v, src_dir, split, cls, force=False, verify_only=False):
    """
    Ensures c23, c40, and cascade variants exist and are valid for a single video.
    Returns dict mapping variant to status ('reused', 'encoded', 'regenerated', 'failed').
    """
    src_path = os.path.join(src_dir, v)
    src_mtime = os.path.getmtime(src_path)
    src_info = get_video_info(src_path)
    if not src_info[0]:
        print(f"  [SOURCE CORRUPT] {src_path}")
        return {"c23": "failed", "c40": "failed", "cascade": "failed"}

    results = {}
    for variant in ["c23", "c40", "cascade"]:
        dst_path = os.path.join(OUTPUT_BASE, variant, split, cls, v)
        already_valid = is_output_valid(dst_path, src_info, src_mtime)

        if already_valid and not force:
            results[variant] = "reused"
            continue

        if verify_only:
            results[variant] = "reused" if already_valid else "failed"
            continue

        was_stale = os.path.exists(dst_path) and not already_valid

        # Execute compression procedure
        success = False
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if variant == "c23":
            try:
                shutil.copy2(src_path, dst_path)
                success = is_output_valid(dst_path, src_info, src_mtime)
            except Exception:
                success = False
        elif variant == "c40":
            success = ffmpeg_encode(src_path, dst_path, crf=40)
            if success:
                success = is_output_valid(dst_path, src_info, src_mtime)
        elif variant == "cascade":
            success = cascade_encode(src_path, dst_path, crf1=23, crf2=40)
            if success:
                success = is_output_valid(dst_path, src_info, src_mtime)

        if success:
            results[variant] = "regenerated" if was_stale else "encoded"
        else:
            results[variant] = "failed"
            if os.path.exists(dst_path):
                os.remove(dst_path)

    return results


def run_pipeline(force=False, sample_limit=None, verify_only=False):
    check_ffmpeg()
    print("=" * 70)
    print("COMPRESSGUARD VIDEO COMPRESSION PIPELINE")
    print(f"  Source  : {SOURCE_BASE} (READ-ONLY)")
    print(f"  Output  : {OUTPUT_BASE}")
    print(f"  Mode    : {'VERIFY-ONLY' if verify_only else ('FORCE REBUILD' if force else 'INCREMENTAL (VALIDATED)')}")
    if sample_limit:
        print(f"  Sample  : First {sample_limit} videos per split/class")
    print("=" * 70)

    stats = {
        "c23":     {"reused": 0, "encoded": 0, "regenerated": 0, "failed": 0},
        "c40":     {"reused": 0, "encoded": 0, "regenerated": 0, "failed": 0},
        "cascade": {"reused": 0, "encoded": 0, "regenerated": 0, "failed": 0},
    }

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = os.path.join(SOURCE_BASE, split, cls)
            if not os.path.isdir(src_dir):
                print(f"[SKIP] Directory missing: {src_dir}")
                continue

            videos = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".mp4")])
            if sample_limit:
                videos = videos[:sample_limit]

            print(f"\nProcessing {split}/{cls}: {len(videos)} videos")

            for i, v in enumerate(videos, 1):
                r = process_video_variants(v, src_dir, split, cls, force=force, verify_only=verify_only)
                for var, outcome in r.items():
                    stats[var][outcome] += 1

                log_str = " | ".join([f"{var}={r[var]}" for var in ["c23", "c40", "cascade"]])
                print(f"  [{i:3d}/{len(videos)}] {v:<15} -> {log_str}")

    # Summary Report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print(f"{'Variant':<10} | {'Reused (Valid)':<15} | {'Freshly Encoded':<16} | {'Regenerated (Stale)':<20} | {'Failed':<8}")
    print("-" * 75)
    for var in ["c23", "c40", "cascade"]:
        s = stats[var]
        print(f"{var:<10} | {s['reused']:<15} | {s['encoded']:<16} | {s['regenerated']:<20} | {s['failed']:<8}")
    print("=" * 70)

    # Representative verification with ffprobe
    print("\nRepresentative Output Verification (FFprobe):")
    for var in ["c23", "c40", "cascade"]:
        sample_path = os.path.join(OUTPUT_BASE, var, "test", "real", "015.mp4")
        if os.path.exists(sample_path):
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=bit_rate,duration", "-of", "default=noprint_wrappers=1", sample_path],
                capture_output=True, text=True
            )
            out = r.stdout.strip().replace("\n", ", ")
            print(f"  {var:<8} test/real/015.mp4: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and validate compression variants.")
    parser.add_argument("--force", action="store_true", help="Force re-encoding even if valid outputs exist.")
    parser.add_argument("--sample", type=int, default=None, help="Process only N videos per split/class for testing.")
    parser.add_argument("--verify-only", action="store_true", help="Audit existing outputs without encoding.")
    args = parser.parse_args()

    run_pipeline(force=args.force, sample_limit=args.sample, verify_only=args.verify_only)
