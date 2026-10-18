import os
import glob
import random
import tensorflow as tf

from model import create_model

FRAME_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\frames"
RESULTS_DIR = r"D:\3rd_yr_sem5\stirs_research\DeepFake\results_v2"

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def collect_frames(split):
    paths = []
    labels = []

    for class_name, label in [("real", 0), ("fake", 1)]:
        class_dir = os.path.join(FRAME_DIR, split, class_name)
        files = glob.glob(os.path.join(class_dir, "*", "*.jpg"))
        paths.extend(files)
        labels.extend([label] * len(files))

    combined = list(zip(paths, labels))
    random.Random(SEED).shuffle(combined)

    paths, labels = zip(*combined)
    return list(paths), list(labels)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)


def create_dataset(paths, labels, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 2048),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_paths, train_labels = collect_frames("train")
    val_paths, val_labels = collect_frames("val")

    print("Training frames:", len(train_paths))
    print("Validation frames:", len(val_paths))

    train_dataset = create_dataset(train_paths, train_labels, shuffle=True)
    val_dataset = create_dataset(val_paths, val_labels)

    model = create_model()
    model.summary()

    checkpoint_path = os.path.join(
        RESULTS_DIR,
        "xception_regularized.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(
        os.path.join(
            RESULTS_DIR,
            "xception_regularized_final.keras"
        )
    )

    with open(os.path.join(RESULTS_DIR, "training_history.txt"), "w") as f:
        for epoch in range(len(history.history["loss"])):
            f.write(
                f"Epoch {epoch + 1}: "
                f"loss={history.history['loss'][epoch]:.4f}, "
                f"accuracy={history.history['accuracy'][epoch]:.4f}, "
                f"auc={history.history['auc'][epoch]:.4f}, "
                f"val_loss={history.history['val_loss'][epoch]:.4f}, "
                f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}, "
                f"val_auc={history.history['val_auc'][epoch]:.4f}\n"
            )

    print("Training completed.")


if __name__ == "__main__":
    main()
