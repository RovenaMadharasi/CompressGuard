import argparse
import glob
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score

FRAME_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\frames"
DEFAULT_MODEL_PATH = r"D:\3rd_yr_sem5\stirs_research\DeepFake\results\xception_baseline.keras"

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def collect_frames():
    paths = []
    labels = []

    for class_name, label in [("real", 0), ("fake", 1)]:
        class_dir = os.path.join(FRAME_DIR, "test", class_name)
        files = glob.glob(os.path.join(class_dir, "*", "*.jpg"))
        paths.extend(files)
        labels.extend([label] * len(files))

    return paths, labels


def load_image(path, label):
    # Model expects raw RGB in [0, 255]; internal model layers perform normalization
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake model on test split.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to .keras model")
    args = parser.parse_args()

    paths, labels = collect_frames()
    if not paths:
        print(f"ERROR: No test frames found in {FRAME_DIR}/test")
        return

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)

    probabilities = model.predict(dataset).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    auc = roc_auc_score(labels, probabilities)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test F1-Score : {f1:.4f}")
    print(f"Test AUC      : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
