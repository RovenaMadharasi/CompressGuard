import argparse
import os
import tempfile

import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from mtcnn import MTCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=0.3446)
    parser.add_argument("--frames", type=int, default=16)
    return parser.parse_args()


def load_model(model_path=None):
    if model_path:
        path = model_path
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Configure it in Streamlit Cloud Secrets."
            )
        path = hf_hub_download(
            repo_id="Rovena28/CompressGuard",
            filename="mixed_cons.keras",
            token=token,
            repo_type="model",
        )

    return tf.keras.models.load_model(
        path,
        compile=False,
        safe_mode=False,
    )


def main():
    args = parse_args()

    min_confidence = 0.85
    margin = 1.30
    size = 299

    cap = cv2.VideoCapture(args.video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise SystemExit("Cannot read video: " + args.video)

    indices = np.linspace(0, frame_count - 1, args.frames).astype(int)
    detector = MTCNN()

    crop_dir = os.path.join(
        tempfile.gettempdir(),
        "compressguard_predict",
        os.path.splitext(os.path.basename(args.video))[0],
    )
    os.makedirs(crop_dir, exist_ok=True)

    def crop_face(frame_bgr):
        faces = detector.detect_faces(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        faces = [
            face for face in faces
            if face["confidence"] >= min_confidence
            and face["box"][2] > 10
            and face["box"][3] > 10
        ]
        if not faces:
            return None

        x, y, width, height = max(
            faces,
            key=lambda face: face["box"][2] * face["box"][3],
        )["box"]

        image_height, image_width = frame_bgr.shape[:2]
        half = max(width, height) * margin / 2.0
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        x1, y1, x2, y2 = [
            int(round(value))
            for value in (
                center_x - half,
                center_y - half,
                center_x + half,
                center_y + half,
            )
        ]

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - image_width)
        pad_bottom = max(0, y2 - image_height)

        if pad_left or pad_top or pad_right or pad_bottom:
            frame_bgr = cv2.copyMakeBorder(
                frame_bgr,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT,
            )
            face_crop = frame_bgr[
                y1 + pad_top:y2 + pad_top,
                x1 + pad_left:x2 + pad_left,
            ]
        else:
            face_crop = frame_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        face_crop = cv2.resize(
            face_crop,
            (size, size),
            interpolation=cv2.INTER_LANCZOS4,
        )
        success, buffer = cv2.imencode(".jpg", face_crop)
        if not success:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    face_crops = []
    for frame_number, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()
        if not success:
            continue

        face_crop = crop_face(frame)
        if face_crop is not None:
            cv2.imwrite(
                os.path.join(crop_dir, f"crop_{frame_number:02d}.jpg"),
                face_crop,
            )
            face_crops.append(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    cap.release()

    print(
        f"Frames sampled: {len(indices)} | faces found: {len(face_crops)} | "
        f"crops saved in {crop_dir}"
    )
    if len(face_crops) < 8:
        print("WARNING: fewer than 8 usable face crops. The result is unreliable.")
    if not face_crops:
        raise SystemExit("No face detected. Cannot classify.")

    model = load_model(args.model)
    probabilities = model.predict(
        np.stack(face_crops).astype("float32"),
        verbose=0,
    ).ravel()
    mean_probability = float(probabilities.mean())

    print("Per-frame fake probability:", np.round(probabilities, 3).tolist())
    print(
        f"Mean fake probability: {mean_probability:.3f} "
        f"(threshold {args.threshold})"
    )
    print(
        "Verdict:",
        "LIKELY FAKE" if mean_probability >= args.threshold else "LIKELY REAL",
    )
    print(
        "This is a probability from a model trained only on FF++ Deepfakes, "
        "not a proof."
    )


if __name__ == "__main__":
    main()
