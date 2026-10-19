
import csv
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("data/raw/cifar-10-batches-py")
OUTPUT = Path("data/processed/cifar10")

PIXEL_COUNT = 3072


def convert_batch(input_path, output_path, batch_name):
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping existing file: {output_path}")
        return

    with input_path.open("rb") as file:
        data = pickle.load(file, encoding="bytes")

    images = np.asarray(data[b"data"], dtype=np.uint8)
    labels = data.get(b"labels", data.get(b"fine_labels"))

    if images.ndim != 2 or images.shape[1] != PIXEL_COUNT:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    if labels is None or len(labels) != len(images):
        raise ValueError("Image and label counts do not match")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")

    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["sample_id", "label"]
                + [f"pixel_{i}" for i in range(PIXEL_COUNT)]
            )

            for index, (image, label) in enumerate(zip(images, labels)):
                normalized = image.astype(np.float32) / 255.0
                writer.writerow(
                    [f"{batch_name}_{index}", int(label)]
                    + normalized.tolist()
                )

        temporary_path.replace(output_path)
        print(f"Created {output_path}: {len(images)} images")

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for batch_number in range(1, 6):
        convert_batch(
            ROOT / f"data_batch_{batch_number}",
            OUTPUT / f"train_batch_{batch_number}.csv",
            f"cifar_train_{batch_number}",
        )

    convert_batch(
        ROOT / "test_batch",
        OUTPUT / "test.csv",
        "cifar_test",
    )

    print("CIFAR-10 conversion finished.")


if __name__ == "__main__":
    main()
