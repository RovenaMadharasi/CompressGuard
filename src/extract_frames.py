import os
import cv2
import numpy as np

DATA_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\data"
FRAME_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\frames"

SPLITS = ["train", "val", "test"]
CLASSES = ["real", "fake"]
FRAMES_PER_VIDEO = 16


def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"No frames found: {video_path}")
        cap.release()
        return

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if success:
            output_file = os.path.join(
                output_dir,
                f"frame_{i + 1:02d}.jpg"
            )
            cv2.imwrite(output_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    cap.release()


def process_dataset():
    for split in SPLITS:
        for class_name in CLASSES:
            input_dir = os.path.join(DATA_DIR, split, class_name)
            output_dir = os.path.join(FRAME_DIR, split, class_name)

            videos = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith(".mp4")
            ]

            print(f"\nProcessing {split}/{class_name}: {len(videos)} videos")

            for index, video in enumerate(videos, start=1):
                video_path = os.path.join(input_dir, video)
                video_name = os.path.splitext(video)[0]
                video_output_dir = os.path.join(output_dir, video_name)

                extract_frames(video_path, video_output_dir)
                print(f"{index}/{len(videos)} - {video}")


if __name__ == "__main__":
    process_dataset()
