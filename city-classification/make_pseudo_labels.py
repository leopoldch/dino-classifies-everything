import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from config import Config
from make_submission import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MONTREAL_SPECIALIST,
    DEFAULT_TTA_RUNS,
    SPECIALIST_CLASSES,
    apply_montreal_specialist,
    build_montreal_specialist_mask,
    default_ensemble_weights,
    default_weight_paths,
    normalize_ensemble_weights,
    predict_logits,
    softmax_np,
)
from utils import (
    detect_model_kind,
    find_test_images,
    load_network_from_weights,
    load_state_dict,
)


MIN_CONFIDENCE = 0.995
MIN_MARGIN = 0.80
MIN_MODEL_CONFIDENCE = 0.90
MAX_PER_CLASS = 500
OUTPUT_CSV = "pseudo_labels_confident.csv"


def top2_margin(probs):
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return top2[:, 1] - top2[:, 0]


def select_confident(final_probs, model_probs, specialist_mask, classes):
    predictions = final_probs.argmax(axis=1)
    confidence = final_probs.max(axis=1)
    margin = top2_margin(final_probs)

    model_preds = np.stack([p.argmax(axis=1) for p in model_probs], axis=1)
    model_min_conf = np.stack([p.max(axis=1) for p in model_probs], axis=1).min(axis=1)
    all_agree = np.all(model_preds == predictions[:, None], axis=1)

    keep = (
        (confidence >= MIN_CONFIDENCE)
        & (margin >= MIN_MARGIN)
        & (model_min_conf >= MIN_MODEL_CONFIDENCE)
        & all_agree
        & ~specialist_mask
    )

    selected = []
    for class_idx in range(len(classes)):
        candidates = np.flatnonzero(keep & (predictions == class_idx))
        candidates = candidates[np.argsort(-confidence[candidates])]
        selected.extend(candidates[:MAX_PER_CLASS].tolist())

    return sorted(selected), predictions, confidence, margin, model_min_conf


def write_csv(selected, test_files, classes, predictions, confidence, margin, model_min_conf):
    csv_path = Path(__file__).resolve().parent / OUTPUT_CSV
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "image_path", "class", "confidence", "margin", "min_model_confidence"])
        for idx in selected:
            writer.writerow([
                test_files[idx].stem,
                str(test_files[idx].resolve()),
                classes[predictions[idx]],
                f"{confidence[idx]:.6f}",
                f"{margin[idx]:.6f}",
                f"{model_min_conf[idx]:.6f}",
            ])
    return csv_path


def main():
    root_dir = Path(__file__).resolve().parent
    config = Config()
    train_dir = config.DATA_DIR / config.COMPETITION / "train"
    test_dir = config.DATA_DIR / config.COMPETITION / "test"

    classes = datasets.ImageFolder(train_dir).classes
    test_files = find_test_images(test_dir)
    if not test_files:
        raise SystemExit(f"No images found in {test_dir}")

    print(f"Test: {len(test_files)} images")
    print(f"Thresholds: conf >= {MIN_CONFIDENCE}, margin >= {MIN_MARGIN}, model_min >= {MIN_MODEL_CONFIDENCE}, max/class = {MAX_PER_CLASS}")

    weight_paths = default_weight_paths(root_dir)
    ensemble_weights = normalize_ensemble_weights(default_ensemble_weights(), len(weight_paths))
    total_weight = float(ensemble_weights.sum())

    weighted_logits = []
    model_probs = []
    for weights_path, ensemble_weight in zip(weight_paths, ensemble_weights):
        kind = detect_model_kind(load_state_dict(weights_path))
        print(f"\n{weights_path.name}, model: {kind}, weight: {ensemble_weight:.2f}")
        network, _, _ = load_network_from_weights(weights_path, len(classes))
        logits = predict_logits(network, test_files, DEFAULT_IMAGE_SIZE, tta=True, tta_runs=DEFAULT_TTA_RUNS)
        weighted_logits.append(ensemble_weight * logits)
        model_probs.append(softmax_np(logits))
        del network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_scores = np.sum(weighted_logits, axis=0) / total_weight

    specialist_mask, specialist_indices, base_probs = build_montreal_specialist_mask(base_scores, classes)
    specialist_path = root_dir / DEFAULT_MONTREAL_SPECIALIST
    print(f"\nSpecialist: {specialist_path.name}, excluded candidates: {int(specialist_mask.sum())}")

    final_probs = base_probs
    if specialist_mask.any():
        specialist_network, _, _ = load_network_from_weights(specialist_path, len(SPECIALIST_CLASSES))
        specialist_files = [test_files[i] for i in np.flatnonzero(specialist_mask)]
        specialist_logits = predict_logits(specialist_network, specialist_files, 384, tta=True, tta_runs=DEFAULT_TTA_RUNS)
        final_probs, _ = apply_montreal_specialist(base_probs, specialist_logits, specialist_indices, specialist_mask)
        del specialist_network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected, predictions, confidence, margin, model_min_conf = select_confident(
        final_probs, model_probs, specialist_mask, classes,
    )
    csv_path = write_csv(selected, test_files, classes, predictions, confidence, margin, model_min_conf)

    counts = Counter(classes[predictions[i]] for i in selected)
    print(f"\nPseudo-labels selected: {len(selected)}")
    for class_name in classes:
        print(f"  {class_name}: {counts[class_name]}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
