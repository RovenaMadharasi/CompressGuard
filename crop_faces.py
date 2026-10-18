"""
crop_faces.py
-------------
Production Face-Cropping Pipeline for FaceForensics++ (c23, c40, cascade).
Features:
  - Detects faces on c23 using MTCNN (confidence >= 0.85).
  - Bounding box: 30% margin, square crop, reflection boundary padding, resized to 299x299.
  - Box detected on c23 is identically applied to c40 and cascade.
  - Fully resumable: checks existing crops and metadata cache.
  - Multi-process worker pool (4 parallel workers) for optimal CPU throughput.
  - Detailed progress logging to outputs/v2/crop_log.txt and console.
  - Comprehensive statistical audit:
      * Split & class retention tables (< 8 frame warning)
      * Cross-split video ID overlap verification
      * Real vs Fake shortcut check (box side, confidence, resolution, file size)
      * Multi-face analysis (>=2 faces with conf >= 0.85)
      * Generates outputs/v2/face_grid_test.png (8 real, 8 fake from test, non-first frames)
      * Generates outputs/v2/multiface_check.png (16 multi-face example crops)
      * Generates outputs/v2/crop_report.json
"""

import os
import sys
import glob
import json
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake"
SRC_FRAMES_BASE = os.path.join(PROJECT_DIR, "frames_compressed")
DST_FRAMES_BASE = os.path.join(PROJECT_DIR, "frames_faces")
OUTPUT_V2_DIR = os.path.join(PROJECT_DIR, "outputs", "v2")
LOG_FILE = os.path.join(OUTPUT_V2_DIR, "crop_log.txt")
METADATA_FILE = os.path.join(OUTPUT_V2_DIR, "crop_metadata.json")

VARIANTS = ["c23", "c40", "cascade"]
SPLITS = ["train", "val", "test"]
CLASSES = ["real", "fake"]
TARGET_SIZE = (299, 299)
MARGIN_FACTOR = 1.30
CONF_THRESHOLD = 0.85
NUM_WORKERS = 4

# Global detector instance for child worker processes
_worker_detector = None

def worker_init():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global _worker_detector
    from mtcnn import MTCNN
    _worker_detector = MTCNN()

def detect_face_box_worker(img_bgr):
    """
    Worker function to detect face using MTCNN.
    Returns: (box, conf, num_faces_ge_85)
    """
    global _worker_detector
    if _worker_detector is None:
        from mtcnn import MTCNN
        _worker_detector = MTCNN()
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = _worker_detector.detect_faces(img_rgb)
    if not faces:
        return None, 0.0, 0
        
    # Count faces with confidence >= CONF_THRESHOLD
    high_conf_faces = [f for f in faces if f.get('confidence', 0.0) >= CONF_THRESHOLD and f['box'][2] >= 20 and f['box'][3] >= 20]
    num_faces_ge_85 = len(high_conf_faces)
    
    if not high_conf_faces:
        return None, 0.0, 0
        
    # Pick largest face
    face = max(high_conf_faces, key=lambda f: f['box'][2] * f['box'][3])
    x, y, w, h = face['box']
    conf = float(face.get('confidence', 0.0))
    
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    side = max(w, h) * MARGIN_FACTOR
    half_side = side / 2.0
    
    x1 = int(round(center_x - half_side))
    y1 = int(round(center_y - half_side))
    x2 = int(round(center_x + half_side))
    y2 = int(round(center_y + half_side))
    
    return (x1, y1, x2, y2, side), conf, num_faces_ge_85

def crop_with_box(img_bgr, box_coords, target_size=TARGET_SIZE):
    x1, y1, x2, y2 = box_coords[:4]
    img_h, img_w, _ = img_bgr.shape
    
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(img_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        crop = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = img_bgr[y1:y2, x1:x2]
        
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
        
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)

def process_single_video(args):
    """
    Task executed by worker process for a single video.
    args: (split, cls, v_name, cached_meta)
    """
    split, cls, v_name, cached_meta = args
    v_c23_dir = os.path.join(SRC_FRAMES_BASE, "c23", split, cls, v_name)
    c23_frames = sorted(glob.glob(os.path.join(v_c23_dir, "*.jpg")))
    
    frame_results = []
    crops_saved = 0
    total_frames = len(c23_frames)
    
    for f_path in c23_frames:
        f_name = os.path.basename(f_path)
        frame_key = f"{split}/{cls}/{v_name}/{f_name}"
        
        # Check if all 3 variant crops already exist
        dst_c23 = os.path.join(DST_FRAMES_BASE, "c23", split, cls, v_name, f_name)
        dst_c40 = os.path.join(DST_FRAMES_BASE, "c40", split, cls, v_name, f_name)
        dst_cas = os.path.join(DST_FRAMES_BASE, "cascade", split, cls, v_name, f_name)
        
        all_exist = (os.path.exists(dst_c23) and os.path.getsize(dst_c23) > 1000 and
                     os.path.exists(dst_c40) and os.path.getsize(dst_c40) > 1000 and
                     os.path.exists(dst_cas) and os.path.getsize(dst_cas) > 1000)
                     
        if all_exist and frame_key in cached_meta:
            # Resumable skip: reuse cached metadata
            meta = cached_meta[frame_key]
            frame_results.append(meta)
            if meta.get("face_detected", False):
                crops_saved += 1
            continue
            
        # Needs processing
        img_c23 = cv2.imread(f_path)
        if img_c23 is None:
            continue
            
        h_orig, w_orig, _ = img_c23.shape
        size_c23 = os.path.getsize(f_path)
        
        box_data, conf, num_faces_ge_85 = detect_face_box_worker(img_c23)
        
        if box_data is None:
            # Detection failed
            meta = {
                "split": split, "class": cls, "video": v_name, "frame": f_name,
                "orig_w": w_orig, "orig_h": h_orig, "orig_bytes": size_c23,
                "face_detected": False, "conf": 0.0, "box_side": 0.0,
                "num_faces_ge_85": 0, "crop_bytes": 0
            }
            frame_results.append(meta)
            continue
            
        # Crop c23
        crop_c23 = crop_with_box(img_c23, box_data)
        if crop_c23 is None:
            continue
            
        # Crop c40
        p_c40 = os.path.join(SRC_FRAMES_BASE, "c40", split, cls, v_name, f_name)
        img_c40 = cv2.imread(p_c40)
        crop_c40 = crop_with_box(img_c40, box_data) if img_c40 is not None else None
        
        # Crop cascade
        p_cas = os.path.join(SRC_FRAMES_BASE, "cascade", split, cls, v_name, f_name)
        img_cas = cv2.imread(p_cas)
        crop_cas = crop_with_box(img_cas, box_data) if img_cas is not None else None
        
        if crop_c40 is None or crop_cas is None:
            continue
            
        # Write crops to disk
        for var, c_img, dst_p in [("c23", crop_c23, dst_c23), ("c40", crop_c40, dst_c40), ("cascade", crop_cas, dst_cas)]:
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            cv2.imwrite(dst_p, c_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        crop_bytes = os.path.getsize(dst_c23)
        crops_saved += 1
        
        meta = {
            "split": split, "class": cls, "video": v_name, "frame": f_name,
            "orig_w": w_orig, "orig_h": h_orig, "orig_bytes": size_c23,
            "face_detected": True, "conf": float(conf), "box_side": float(box_data[4]),
            "num_faces_ge_85": int(num_faces_ge_85), "crop_bytes": crop_bytes
        }
        frame_results.append(meta)
        
    return split, cls, v_name, frame_results, crops_saved, total_frames

def log_msg(log_f, msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()

def main():
    os.makedirs(DST_FRAMES_BASE, exist_ok=True)
    os.makedirs(OUTPUT_V2_DIR, exist_ok=True)
    
    log_f = open(LOG_FILE, "a", encoding="utf-8")
    log_msg(log_f, "==============================================================================")
    log_msg(log_f, "STARTING FULL COMPRESSGUARD FACE CROPPING PIPELINE")
    log_msg(log_f, f"  Workers: {NUM_WORKERS} parallel processes | MTCNN Conf >= {CONF_THRESHOLD} | Margin: 30%")
    log_msg(log_f, "==============================================================================")
    
    # Load cached metadata if available (for resumability)
    cached_metadata = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                cached_metadata = json.load(f)
            log_msg(log_f, f"Loaded {len(cached_metadata)} cached frame metadata entries from {METADATA_FILE}")
        except Exception as e:
            log_msg(log_f, f"Warning: Failed to load cached metadata: {e}")
            
    # Gather all video tasks
    tasks = []
    total_videos_found = 0
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = os.path.join(SRC_FRAMES_BASE, "c23", split, cls)
            if not os.path.isdir(cls_dir):
                continue
            videos = sorted(os.listdir(cls_dir))
            total_videos_found += len(videos)
            for v in videos:
                tasks.append((split, cls, v, cached_metadata))
                
    log_msg(log_f, f"Total videos queued for processing across all splits: {len(tasks)}")
    
    start_time = time.time()
    all_metadata = dict(cached_metadata)
    processed_count = 0
    total_tasks = len(tasks)
    
    # Process tasks using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=worker_init) as executor:
        future_to_task = {executor.submit(process_single_video, t): t for t in tasks}
        
        for future in as_completed(future_to_task):
            processed_count += 1
            try:
                split, cls, v_name, frame_results, crops_saved, total_frames = future.result()
                for meta in frame_results:
                    key = f"{meta['split']}/{meta['class']}/{meta['video']}/{meta['frame']}"
                    all_metadata[key] = meta
                    
                flag_str = " [WARNING: < 8 frames!]" if crops_saved < 8 else ""
                log_msg(log_f, f"[{processed_count:3d}/{total_tasks:3d}] ({split.upper():<5s}/{cls:<4s}) {v_name:<16s}: {crops_saved:2d}/{total_frames:2d} faces cropped{flag_str}")
                
                # Periodically save metadata
                if processed_count % 25 == 0 or processed_count == total_tasks:
                    with open(METADATA_FILE, "w") as f:
                        json.dump(all_metadata, f, indent=2)
                        
            except Exception as e:
                task_info = future_to_task[future]
                log_msg(log_f, f"[ERROR] Task failed for {task_info[:3]}: {e}")
                
    elapsed_total = time.time() - start_time
    log_msg(log_f, f"All video tasks finished in {elapsed_total:.2f}s ({elapsed_total/60:.2f} mins).")
    
    # Save final metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)
        
    # -------------------------------------------------------------------------
    # COMPUTE AUDIT METRICS & STATISTICAL REPORTS
    # -------------------------------------------------------------------------
    log_msg(log_f, "\n" + "=" * 78)
    log_msg(log_f, "COMPUTING POST-CROP STATISTICAL AUDIT & VERIFICATION")
    log_msg(log_f, "=" * 78)
    
    # 1. Retention per split and class
    retention_table = {}
    video_retention = {} # split -> cls -> dict(video -> count)
    low_frame_videos = []
    
    for split in SPLITS:
        retention_table[split] = {}
        video_retention[split] = {}
        for cls in CLASSES:
            # Source frames count
            src_pattern = os.path.join(SRC_FRAMES_BASE, "c23", split, cls, "*", "*.jpg")
            src_files = glob.glob(src_pattern)
            before_cnt = len(src_files)
            
            # Destination crops count
            dst_pattern = os.path.join(DST_FRAMES_BASE, "c23", split, cls, "*", "*.jpg")
            dst_files = glob.glob(dst_pattern)
            after_cnt = len(dst_files)
            
            # Count per video
            v_dirs = sorted(os.listdir(os.path.join(SRC_FRAMES_BASE, "c23", split, cls)))
            n_vids = len(v_dirs)
            
            video_counts = {}
            for v in v_dirs:
                v_crops = len(glob.glob(os.path.join(DST_FRAMES_BASE, "c23", split, cls, v, "*.jpg")))
                video_counts[v] = v_crops
                if v_crops < 8:
                    low_frame_videos.append({"split": split, "class": cls, "video": v, "valid_frames": v_crops, "total_frames": 16})
                    
            video_retention[split][cls] = video_counts
            pct = (after_cnt / before_cnt * 100) if before_cnt > 0 else 0
            retention_table[split][cls] = {
                "videos": n_vids,
                "frames_before": before_cnt,
                "frames_after": after_cnt,
                "retention_pct": round(pct, 2),
                "avg_frames_per_video": round(after_cnt / max(1, n_vids), 2)
            }
            
    # Print retention table
    log_msg(log_f, "\n1. Split & Class Retention Table:")
    log_msg(log_f, f"{'Split':<8} | {'Class':<6} | {'Videos':<8} | {'Frames Before':<14} | {'Frames After':<14} | {'Retention %':<12} | {'Avg/Video'}")
    log_msg(log_f, "-" * 82)
    for s in SPLITS:
        for c in CLASSES:
            r = retention_table[s][c]
            log_msg(log_f, f"{s.upper():<8} | {c:<6} | {r['videos']:<8} | {r['frames_before']:<14} | {r['frames_after']:<14} | {r['retention_pct']:<10.2f}% | {r['avg_frames_per_video']:<6.2f}")
    log_msg(log_f, "-" * 82)
    
    if low_frame_videos:
        log_msg(log_f, f"\n[ALERT] {len(low_frame_videos)} videos have fewer than 8 valid frames:")
        for l in low_frame_videos:
            log_msg(log_f, f"  - {l['split']}/{l['class']}/{l['video']}: {l['valid_frames']}/{l['total_frames']} frames")
    else:
        log_msg(log_f, "\n[OK] Zero videos have fewer than 8 valid frames.")
        
    # 2. Overlap & Leakage Check on frames_faces/
    log_msg(log_f, "\n2. Cross-Split Overlap & Leakage Check on frames_faces/:")
    split_ids = {s: set() for s in SPLITS}
    for s in SPLITS:
        for c in CLASSES:
            d = os.path.join(DST_FRAMES_BASE, "c23", s, c)
            if os.path.isdir(d):
                for v in os.listdir(d):
                    if "_" in v:
                        t, src = v.split("_")
                        split_ids[s].add(t)
                        split_ids[s].add(src)
                    else:
                        split_ids[s].add(v)
                        
    leak_tv = len(split_ids["train"] & split_ids["val"])
    leak_tt = len(split_ids["train"] & split_ids["test"])
    leak_vt = len(split_ids["val"] & split_ids["test"])
    
    log_msg(log_f, f"  Unique Actor IDs in Train : {len(split_ids['train'])}")
    log_msg(log_f, f"  Unique Actor IDs in Val   : {len(split_ids['val'])}")
    log_msg(log_f, f"  Unique Actor IDs in Test  : {len(split_ids['test'])}")
    log_msg(log_f, f"  Overlap Train vs Val      : {leak_tv} (PASSED)" if leak_tv == 0 else f"  Overlap Train vs Val: {leak_tv} (FAILED)")
    log_msg(log_f, f"  Overlap Train vs Test     : {leak_tt} (PASSED)" if leak_tt == 0 else f"  Overlap Train vs Test: {leak_tt} (FAILED)")
    log_msg(log_f, f"  Overlap Val vs Test       : {leak_vt} (PASSED)" if leak_vt == 0 else f"  Overlap Val vs Test: {leak_vt} (FAILED)")
    
    # 3. Shortcut Check: Real vs Fake per split
    log_msg(log_f, "\n3. Real vs Fake Shortcut Check:")
    shortcut_report = {}
    
    for s in SPLITS:
        shortcut_report[s] = {}
        log_msg(log_f, f"\n  --- SPLIT: {s.upper()} ---")
        
        # Collect detected frame records for real and fake
        real_records = [m for m in all_metadata.values() if m['split'] == s and m['class'] == 'real' and m.get('face_detected', False)]
        fake_records = [m for m in all_metadata.values() if m['split'] == s and m['class'] == 'fake' and m.get('face_detected', False)]
        
        # Face box side length
        real_sides = [m['box_side'] for m in real_records]
        fake_sides = [m['box_side'] for m in fake_records]
        mean_side_r, std_side_r = (np.mean(real_sides), np.std(real_sides)) if real_sides else (0, 0)
        mean_side_f, std_side_f = (np.mean(fake_sides), np.std(fake_sides)) if fake_sides else (0, 0)
        diff_side_pct = abs(mean_side_f - mean_side_r) / mean_side_r * 100 if mean_side_r > 0 else 0
        
        # Detector confidence
        real_confs = [m['conf'] for m in real_records]
        fake_confs = [m['conf'] for m in fake_records]
        mean_conf_r, std_conf_r = (np.mean(real_confs), np.std(real_confs)) if real_confs else (0, 0)
        mean_conf_f, std_conf_f = (np.mean(fake_confs), np.std(fake_confs)) if fake_confs else (0, 0)
        diff_conf_pct = abs(mean_conf_f - mean_conf_r) / mean_conf_r * 100 if mean_conf_r > 0 else 0
        
        # Resolution distributions
        res_real = {}
        for m in real_records:
            res_str = f"{m['orig_w']}x{m['orig_h']}"
            res_real[res_str] = res_real.get(res_str, 0) + 1
        res_fake = {}
        for m in fake_records:
            res_str = f"{m['orig_w']}x{m['orig_h']}"
            res_fake[res_str] = res_fake.get(res_str, 0) + 1
            
        # File sizes
        real_bytes = [m['orig_bytes'] for m in real_records]
        fake_bytes = [m['orig_bytes'] for m in fake_records]
        mean_bytes_r, std_bytes_r = (np.mean(real_bytes), np.std(real_bytes)) if real_bytes else (0, 0)
        mean_bytes_f, std_bytes_f = (np.mean(fake_bytes), np.std(fake_bytes)) if fake_bytes else (0, 0)
        diff_bytes_pct = abs(mean_bytes_f - mean_bytes_r) / mean_bytes_r * 100 if mean_bytes_r > 0 else 0
        
        # Crop file sizes
        real_crop_bytes = [m['crop_bytes'] for m in real_records]
        fake_crop_bytes = [m['crop_bytes'] for m in fake_records]
        mean_crop_r = np.mean(real_crop_bytes) if real_crop_bytes else 0
        mean_crop_f = np.mean(fake_crop_bytes) if fake_crop_bytes else 0
        diff_crop_pct = abs(mean_crop_f - mean_crop_r) / mean_crop_r * 100 if mean_crop_r > 0 else 0
        
        # Flags
        flags = []
        if diff_side_pct > 10.0:
            flags.append(f"Face box side difference > 10% ({diff_side_pct:.1f}%)")
        if diff_conf_pct > 10.0:
            flags.append(f"Confidence difference > 10% ({diff_conf_pct:.1f}%)")
        if diff_bytes_pct > 10.0:
            flags.append(f"Original file size difference > 10% ({diff_bytes_pct:.1f}%)")
        if diff_crop_pct > 10.0:
            flags.append(f"Crop file size difference > 10% ({diff_crop_pct:.1f}%)")
            
        log_msg(log_f, f"    Face Box Side  : Real = {mean_side_r:.1f} ± {std_side_r:.1f} px | Fake = {mean_side_f:.1f} ± {std_side_f:.1f} px (Diff: {diff_side_pct:.2f}%)")
        log_msg(log_f, f"    MTCNN Conf     : Real = {mean_conf_r:.4f} ± {std_conf_r:.4f} | Fake = {mean_conf_f:.4f} ± {std_conf_f:.4f} (Diff: {diff_conf_pct:.2f}%)")
        log_msg(log_f, f"    Orig File Size : Real = {mean_bytes_r/1024:.1f} KB | Fake = {mean_bytes_f/1024:.1f} KB (Diff: {diff_bytes_pct:.2f}%)")
        log_msg(log_f, f"    Crop File Size : Real = {mean_crop_r/1024:.1f} KB | Fake = {mean_crop_f/1024:.1f} KB (Diff: {diff_crop_pct:.2f}%)")
        log_msg(log_f, f"    Resolutions    : Real = {res_real} | Fake = {res_fake}")
        if flags:
            log_msg(log_f, f"    [FLAGGED SHORTCUTS] {'; '.join(flags)}")
        else:
            log_msg(log_f, "    [OK] No shortcut differences > 10% detected.")
            
        shortcut_report[s] = {
            "face_box_side": {"real_mean": round(mean_side_r, 2), "real_std": round(std_side_r, 2), "fake_mean": round(mean_side_f, 2), "fake_std": round(std_side_f, 2), "diff_pct": round(diff_side_pct, 2)},
            "confidence": {"real_mean": round(mean_conf_r, 4), "real_std": round(std_conf_r, 4), "fake_mean": round(mean_conf_f, 4), "fake_std": round(std_conf_f, 4), "diff_pct": round(diff_conf_pct, 2)},
            "original_file_size_kb": {"real_mean": round(mean_bytes_r/1024, 2), "fake_mean": round(mean_bytes_f/1024, 2), "diff_pct": round(diff_bytes_pct, 2)},
            "crop_file_size_kb": {"real_mean": round(mean_crop_r/1024, 2), "fake_mean": round(mean_crop_f/1024, 2), "diff_pct": round(diff_crop_pct, 2)},
            "resolutions": {"real": res_real, "fake": res_fake},
            "flagged_shortcuts": flags
        }
        
    # 4. Multi-face Analysis
    log_msg(log_f, "\n4. Multi-face Detection Analysis (>= 2 faces with conf >= 0.85):")
    multiface_report = {}
    multiface_candidates = []
    
    for s in SPLITS:
        multiface_report[s] = {}
        for c in CLASSES:
            records = [m for m in all_metadata.values() if m['split'] == s and m['class'] == c]
            multi_cnt = sum(1 for m in records if m.get('num_faces_ge_85', 0) >= 2)
            total_cnt = len(records)
            pct = (multi_cnt / total_cnt * 100) if total_cnt > 0 else 0
            multiface_report[s][c] = {"multi_face_frames": multi_cnt, "total_frames": total_cnt, "pct": round(pct, 2)}
            log_msg(log_f, f"  {s.upper():<5s} / {c:<4s}: {multi_cnt:3d} / {total_cnt:4d} frames ({pct:5.2f}%) have >= 2 high-confidence faces")
            
            for m in records:
                if m.get('num_faces_ge_85', 0) >= 2:
                    crop_p = os.path.join(DST_FRAMES_BASE, "c23", s, c, m['video'], m['frame'])
                    if os.path.exists(crop_p):
                        multiface_candidates.append((crop_p, s, c, m['video'], m['frame'], m['num_faces_ge_85']))
                        
    # Save 16 example crops of multi-face frames
    log_msg(log_f, f"\nSaving 16 multi-face example crops to {os.path.join(OUTPUT_V2_DIR, 'multiface_check.png')}...")
    if len(multiface_candidates) >= 16:
        random.seed(42)
        sample_multi = random.sample(multiface_candidates, 16)
    else:
        sample_multi = multiface_candidates
        
    if sample_multi:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle("Multi-Face Frames (Largest Face Cropped) - 16 Examples", fontsize=15, fontweight="bold")
        for idx in range(16):
            r, c = idx // 4, idx % 4
            ax = axes[r, c]
            if idx < len(sample_multi):
                p, s, cls_name, v, f, n_faces = sample_multi[idx]
                bgr = cv2.imread(p)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                color = "green" if cls_name == "real" else "red"
                ax.imshow(rgb)
                ax.set_title(f"[{cls_name.upper()}] {v}/{f}\n({n_faces} faces detected)", fontsize=9, color=color, fontweight="bold")
            ax.axis("off")
        plt.tight_layout()
        multi_out = os.path.join(OUTPUT_V2_DIR, "multiface_check.png")
        plt.savefig(multi_out, dpi=150)
        plt.close()
        log_msg(log_f, f"  [SUCCESS] Saved: {multi_out}")
    else:
        log_msg(log_f, "  No multi-face candidate crops available to save.")
        
    # 5. Save outputs/v2/face_grid_test.png: 8 real and 8 fake tiles from TEST split
    # Each from a DIFFERENT video, from DIFFERENT frame positions (not frame_01)
    log_msg(log_f, "\n5. Generating outputs/v2/face_grid_test.png (8 real, 8 fake from TEST split, non-first frames)...")
    test_real_vids = sorted(os.listdir(os.path.join(DST_FRAMES_BASE, "c23", "test", "real")))
    test_fake_vids = sorted(os.listdir(os.path.join(DST_FRAMES_BASE, "c23", "test", "fake")))
    
    random.seed(42)
    sel_real_vids = random.sample(test_real_vids, min(8, len(test_real_vids)))
    sel_fake_vids = random.sample(test_fake_vids, min(8, len(test_fake_vids)))
    
    test_tiles = []
    
    # Select non-first frame for real videos
    for v in sel_real_vids:
        frames = sorted(glob.glob(os.path.join(DST_FRAMES_BASE, "c23", "test", "real", v, "*.jpg")))
        non_first = [f for f in frames if not f.endswith("frame_01.jpg")]
        chosen = random.choice(non_first) if non_first else (frames[0] if frames else None)
        if chosen:
            test_tiles.append((chosen, "Real (0)", v, os.path.basename(chosen)))
            
    # Select non-first frame for fake videos
    for v in sel_fake_vids:
        frames = sorted(glob.glob(os.path.join(DST_FRAMES_BASE, "c23", "test", "fake", v, "*.jpg")))
        non_first = [f for f in frames if not f.endswith("frame_01.jpg")]
        chosen = random.choice(non_first) if non_first else (frames[0] if frames else None)
        if chosen:
            test_tiles.append((chosen, "Fake (1)", v, os.path.basename(chosen)))
            
    random.shuffle(test_tiles)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("CompressGuard Test Set Face Crops (Non-First Frames) - 4x4 Grid", fontsize=15, fontweight="bold")
    for idx, (path, label_str, v_name, f_name) in enumerate(test_tiles[:16]):
        r, c = idx // 4, idx % 4
        ax = axes[r, c]
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        color = "green" if "Real" in label_str else "red"
        ax.imshow(rgb)
        ax.set_title(f"{label_str}\n[{v_name}/{f_name}]\n299x299", fontsize=9, color=color, fontweight="bold")
        ax.axis("off")
        
    plt.tight_layout()
    test_grid_out = os.path.join(OUTPUT_V2_DIR, "face_grid_test.png")
    plt.savefig(test_grid_out, dpi=150)
    plt.close()
    log_msg(log_f, f"  [SUCCESS] Saved: {test_grid_out}")
    
    # 6. Save comprehensive JSON summary
    report_data = {
        "summary": {
            "total_videos": total_tasks,
            "pipeline_elapsed_seconds": round(elapsed_total, 2),
            "workers_used": NUM_WORKERS,
            "variants_processed": VARIANTS,
            "crop_spec": {
                "detector": "MTCNN",
                "min_confidence": CONF_THRESHOLD,
                "margin_pct": 30,
                "target_size": [299, 299],
                "box_reuse": "Detected on c23, identically applied to c40 and cascade"
            }
        },
        "retention_by_split": retention_table,
        "low_frame_videos": low_frame_videos,
        "cross_split_leakage": {
            "train_val_overlap": leak_tv,
            "train_test_overlap": leak_tt,
            "val_test_overlap": leak_vt,
            "status": "PASSED (Zero Leakage)" if (leak_tv == 0 and leak_tt == 0 and leak_vt == 0) else "FAILED"
        },
        "shortcut_check": shortcut_report,
        "multiface_check": multiface_report,
        "output_artifacts": {
            "face_crops_directory": DST_FRAMES_BASE,
            "metadata_file": METADATA_FILE,
            "crop_log": LOG_FILE,
            "face_grid_test": test_grid_out,
            "multiface_check_grid": os.path.join(OUTPUT_V2_DIR, "multiface_check.png"),
            "report_json": os.path.join(OUTPUT_V2_DIR, "crop_report.json")
        }
    }
    
    report_json_path = os.path.join(OUTPUT_V2_DIR, "crop_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    log_msg(log_f, f"  [SUCCESS] Saved comprehensive audit report to: {report_json_path}")
    log_msg(log_f, "\nFACE CROPPING PIPELINE AND AUDIT COMPLETE.")
    log_f.close()

if __name__ == "__main__":
    main()
