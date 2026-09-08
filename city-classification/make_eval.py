import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parent

from config import Config
from utils import (
    AUGMENT_SUFFIXES,
    DEVICE,
    SEED,
    build_eval_transform,
    build_validation_dataset,
    find_weight_files,
    infer_image_size,
    load_network_from_weights,
    split_by_base_image,
)


class RemappedSubset(Dataset):
    def __init__(self, root_dir, indices, transform, class_names):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.indices = list(indices)
        self.label_mapping = {
            self.dataset.class_to_idx[class_name]: index
            for index, class_name in enumerate(class_names)
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, self.label_mapping[label]


def build_specialist_validation_dataset(image_size, batch_size, specialist_classes):
    config = Config()
    train_dir = config.DATA_DIR / config.COMPETITION / "train"
    full_dataset = datasets.ImageFolder(train_dir)

    missing_classes = [class_name for class_name in specialist_classes if class_name not in full_dataset.class_to_idx]
    if missing_classes:
        raise ValueError(f"Classes not found in dataset: {missing_classes}")

    selected_indices = [
        index
        for index, (_, label) in enumerate(full_dataset.samples)
        if full_dataset.classes[label] in specialist_classes
    ]
    selected_samples = [full_dataset.samples[index] for index in selected_indices]
    _, val_pos = split_by_base_image(selected_samples, SEED, 0.1, AUGMENT_SUFFIXES)
    val_indices = [selected_indices[index] for index in val_pos]

    val_dataset = RemappedSubset(
        train_dir,
        val_indices,
        build_eval_transform(image_size),
        specialist_classes,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    return val_loader, specialist_classes, full_dataset, val_indices


def run_inference(network, loader, num_classes: int):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    confidences = []
    errors = []  # (dataset_index, true_label, pred_label)

    network.to(DEVICE).eval()
    sample_idx = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = network(images.to(DEVICE))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for i in range(len(labels_np)):
                true = labels_np[i]
                pred = probs[i].argmax()
                conf = probs[i].max()
                confidences.append(conf)
                matrix[true, pred] += 1
                if pred != true:
                    errors.append((sample_idx + i, true, pred))

            sample_idx += len(labels_np)

    return matrix, confidences, errors


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            value = normalized[i, j]
            text = f"{count}\n{value:.2f}"
            color = "white" if value > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confidence_histogram(confidences: list[float], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=20, range=(0, 1), edgecolor="black")
    ax.set_xlabel("Confidence (max softmax prob)")
    ax.set_ylabel("Number of images")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def save_error_images(errors: list[tuple], val_idx: list[int], full_dataset, class_names: list[str], errors_dir: Path):
    errors_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx, true, pred in errors:
        real_idx = val_idx[sample_idx]
        img_path, _ = full_dataset.samples[real_idx]
        img = Image.open(img_path).convert("RGB")
        fname = f"true_{class_names[true]}__pred_{class_names[pred]}__{Path(img_path).name}"
        dest = errors_dir / fname
        img.save(dest)
    print(f"  saved {len(errors)} error images in {errors_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="*", help="Checkpoint(s). Default: weights/*.pt|.pth|.p")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--specialist-classes",
        nargs="+",
        default=None,
        help="Evaluate a specialist checkpoint on a class subset, e.g. Montreal Quebec Boston",
    )
    args = parser.parse_args()

    weight_files = find_weight_files(ROOT, args.weights, include_legacy=True)
    if not weight_files:
        raise SystemExit("No checkpoints found.")

    for weights_path in weight_files:
        print(f"\n{weights_path.name}")
        image_size = infer_image_size(weights_path, args.image_size)

        if args.specialist_classes:
            loader, class_names, full_dataset, val_idx = build_specialist_validation_dataset(
                image_size,
                args.batch_size,
                args.specialist_classes,
            )
        else:
            loader, class_names, full_dataset, val_idx = build_validation_dataset(image_size, args.batch_size)

        network, kind, _ = load_network_from_weights(weights_path, len(class_names))
        print(f"  detected model: {kind}, image_size: {image_size}px")
        if args.specialist_classes:
            print(f"  specialist mode: {class_names}")

        matrix, confidences, errors = run_inference(network, loader, len(class_names))

        stem = weights_path.stem
        artefacts = ROOT / "artefacts"

        cm_path = artefacts / f"{stem}_confusion_matrix.png"
        save_confusion_matrix(matrix, class_names, cm_path, f"{weights_path.name} ({kind}, {image_size}px)")
        print(f"  saved {cm_path}")

        hist_path = artefacts / f"{stem}_confidence_hist.png"
        save_confidence_histogram(confidences, hist_path, f"Confidence - {weights_path.name}")

        errors_dir = artefacts / "errors" / stem
        save_error_images(errors, list(val_idx), full_dataset, class_names, errors_dir)

        acc = matrix.diagonal().sum() / matrix.sum()
        print(f"  val accuracy: {acc:.3f} | errors: {len(errors)}/{len(confidences)}")


if __name__ == "__main__":
    main()
