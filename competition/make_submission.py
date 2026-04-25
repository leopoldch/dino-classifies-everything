import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from config import Config
from utils import (
    DEFAULT_TTA_RUNS,
    DEVICE,
    TestDataset,
    detect_model_kind,
    find_test_images,
    find_weight_files,
    get_prediction_transforms,
    infer_image_size,
    load_network_from_weights,
    load_state_dict,
    write_submission_csv,
)

BATCH_SIZE = 16
SPECIALIST_CLASSES = ("Montreal", "Quebec", "Boston")
DEFAULT_ENSEMBLE = (
    ("weights/dinov3-evolved-final.pt", 0.20),
    ("weights/dinov2-evolved-final-dernier.pt", 0.40),
    ("weights/dinov3-gem-final.pt", 0.40),
)
DEFAULT_IMAGE_SIZE = 392
DEFAULT_MONTREAL_SPECIALIST = "weights/montreal-specialist-evolved-final.pt"
MONTREAL_SPECIALIST_ALPHA = 0.60
MONTREAL_SPECIALIST_MAX_MARGIN = 0.08
MONTREAL_SPECIALIST_MIN_MASS = 0.55


def normalize_ensemble_weights(raw_weights: list[float] | None, num_models: int) -> np.ndarray:
    if raw_weights is None:
        return np.ones(num_models, dtype=np.float32)
    if len(raw_weights) == 1 and num_models > 1:
        raw_weights = raw_weights * num_models
    if len(raw_weights) != num_models:
        raise ValueError("Fournir soit une seule pondération, soit une pondération par checkpoint.")

    weights = np.asarray(raw_weights, dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError("Les pondérations doivent être positives ou nulles.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("La somme des pondérations doit être strictement positive.")
    return weights


def default_weight_paths(root_dir: Path) -> list[Path]:
    return [root_dir / path for path, _ in DEFAULT_ENSEMBLE]


def default_ensemble_weights() -> list[float]:
    return [weight for _, weight in DEFAULT_ENSEMBLE]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def build_montreal_specialist_mask(
    base_logits: np.ndarray,
    classes: list[str],
    specialist_classes: tuple[str, ...] = SPECIALIST_CLASSES,
    max_margin: float = MONTREAL_SPECIALIST_MAX_MARGIN,
    min_mass: float = MONTREAL_SPECIALIST_MIN_MASS,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    if max_margin < 0.0:
        raise ValueError("La marge maximale du spécialiste doit être positive ou nulle.")
    if not 0.0 <= min_mass <= 1.0:
        raise ValueError("La masse minimale du spécialiste doit être entre 0 et 1.")

    specialist_indices = [classes.index(class_name) for class_name in specialist_classes]
    base_probs = softmax_np(base_logits)
    ranked = np.argsort(base_probs, axis=1)[:, ::-1]

    top1 = ranked[:, 0]
    top2 = ranked[:, 1]
    top2_in_specialist = np.isin(top1, specialist_indices) & np.isin(top2, specialist_indices)

    row_indices = np.arange(base_probs.shape[0])
    margin = base_probs[row_indices, top1] - base_probs[row_indices, top2]
    specialist_mass = base_probs[:, specialist_indices].sum(axis=1)

    apply_mask = top2_in_specialist & (margin <= max_margin) & (specialist_mass >= min_mass)
    return apply_mask, specialist_indices, base_probs


def apply_montreal_specialist(
    base_probs: np.ndarray,
    specialist_logits: np.ndarray,
    specialist_indices: list[int],
    apply_mask: np.ndarray,
    alpha: float = MONTREAL_SPECIALIST_ALPHA,
) -> tuple[np.ndarray, int]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Le poids du spécialiste doit être entre 0 et 1.")

    specialist_probs = softmax_np(specialist_logits)
    combined_probs = base_probs.copy()
    mask_indices = np.flatnonzero(apply_mask)
    if len(mask_indices) == 0:
        return combined_probs, 0

    specialist_slice = base_probs[mask_indices][:, specialist_indices]
    specialist_mass = specialist_slice.sum(axis=1, keepdims=True)
    normalized_base_slice = np.divide(
        specialist_slice,
        specialist_mass,
        out=np.full_like(specialist_slice, 1.0 / len(specialist_indices)),
        where=specialist_mass > 0,
    )
    blended_slice = (1.0 - alpha) * normalized_base_slice + alpha * specialist_probs
    blended_slice /= blended_slice.sum(axis=1, keepdims=True)
    combined_probs[mask_indices[:, None], specialist_indices] = specialist_mass * blended_slice

    return combined_probs, len(mask_indices)


def predict_logits(
    network,
    test_files: list[Path],
    image_size: int,
    tta: bool,
    tta_runs: int,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> np.ndarray:
    network.to(device).eval()
    all_transforms = get_prediction_transforms(image_size, tta=tta, tta_runs=tta_runs)

    all_logits = []
    with torch.no_grad():
        for transform in all_transforms:
            dataset = TestDataset(test_files, transform)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            batch_logits = []
            for batch in loader:
                logits = network(batch.to(device))
                batch_logits.append(logits.cpu().numpy())
            all_logits.append(np.concatenate(batch_logits, axis=0))

    return np.mean(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Génère submission.csv depuis un ou plusieurs checkpoints.")
    parser.add_argument("weights", nargs="*", help="Chemins vers les .pt. Défaut: l'ensemble hardcodé.")
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs="+",
        help="Pondérations de l'ensemble, dans le même ordre que les checkpoints.",
    )
    parser.add_argument(
        "--montreal-specialist",
        default=None,
        help="Checkpoint spécialiste 3 classes pour Montréal/Québec/Boston.",
    )
    parser.add_argument(
        "--no-montreal-specialist",
        action="store_true",
        help="Désactive le spécialiste hardcodé.",
    )
    parser.add_argument(
        "--montreal-specialist-image-size",
        type=int,
        default=None,
        help="Forcer la taille d'image du spécialiste.",
    )
    parser.add_argument("--image-size", type=int, default=None,
                        help="Forcer la taille d'image (sinon auto-détectée depuis le nom de fichier)")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--tta-runs", type=int, default=DEFAULT_TTA_RUNS)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    config = Config()
    train_dir = config.DATA_DIR / config.COMPETITION / "train"
    test_dir = config.DATA_DIR / config.COMPETITION / "test"

    classes = datasets.ImageFolder(train_dir).classes
    test_files = find_test_images(test_dir)

    if not test_files:
        raise SystemExit(f"Aucune image trouvée dans {test_dir}")
    print(f"Test: {len(test_files)} images trouvées ({test_files[0].suffix})")

    root_dir = Path(__file__).resolve().parent
    using_default_ensemble = not args.weights
    weight_paths = (
        default_weight_paths(root_dir)
        if using_default_ensemble
        else find_weight_files(root_dir, args.weights)
    )
    if not weight_paths:
        raise SystemExit("Aucun checkpoint trouvé.")
    if using_default_ensemble:
        print(f"Ensemble hardcodé: {[path.name for path in weight_paths]}")

    try:
        raw_weights = args.ensemble_weights
        if raw_weights is None and using_default_ensemble:
            raw_weights = default_ensemble_weights()
        ensemble_weights = normalize_ensemble_weights(raw_weights, len(weight_paths))
    except ValueError as exc:
        parser.error(str(exc))

    weight_infos = []
    for weights_path, ensemble_weight in zip(weight_paths, ensemble_weights):
        kind = detect_model_kind(load_state_dict(weights_path))
        image_size = infer_image_size(
            weights_path,
            args.image_size if args.image_size is not None else DEFAULT_IMAGE_SIZE,
        )
        weight_infos.append((weights_path, kind, image_size, float(ensemble_weight)))
        print(f"\n{weights_path.name}")
        print(f"  modele: {kind}, image_size: {image_size}px, TTA: {not args.no_tta}, poids_ensemble: {ensemble_weight:.4f}")

    total_weight = float(ensemble_weights.sum())
    all_weighted_logits = []

    for weights_path, _, image_size, ensemble_weight in weight_infos:
        network, _, _ = load_network_from_weights(weights_path, len(classes))
        logits = predict_logits(
            network,
            test_files,
            image_size,
            tta=not args.no_tta,
            tta_runs=args.tta_runs,
        )
        all_weighted_logits.append(ensemble_weight * logits)

        del network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    submission_scores = np.sum(all_weighted_logits, axis=0) / total_weight

    specialist_arg = None
    if not args.no_montreal_specialist:
        specialist_arg = args.montreal_specialist or str(root_dir / DEFAULT_MONTREAL_SPECIALIST)

    if specialist_arg:
        apply_mask, specialist_indices, submission_probs = build_montreal_specialist_mask(
            submission_scores,
            classes,
            max_margin=MONTREAL_SPECIALIST_MAX_MARGIN,
            min_mass=MONTREAL_SPECIALIST_MIN_MASS,
        )
        specialist_count = int(apply_mask.sum())
        specialist_path = Path(specialist_arg)
        specialist_kind = detect_model_kind(load_state_dict(specialist_path))
        specialist_image_size = infer_image_size(
            specialist_path,
            args.montreal_specialist_image_size,
        )
        print(f"\nSpécialiste Montréal actif: {specialist_path.name}")
        print(f"  modele: {specialist_kind}, image_size: {specialist_image_size}px, alpha: {MONTREAL_SPECIALIST_ALPHA:.2f}")
        print(f"  gating: top-2 dans le trio, marge <= {MONTREAL_SPECIALIST_MAX_MARGIN:.2f}, masse trio >= {MONTREAL_SPECIALIST_MIN_MASS:.2f}")
        print(f"  images candidates: {specialist_count}")

        if specialist_count > 0:
            specialist_network, _, _ = load_network_from_weights(
                specialist_path,
                len(SPECIALIST_CLASSES),
            )
            candidate_files = [test_files[index] for index in np.flatnonzero(apply_mask)]
            specialist_logits = predict_logits(
                specialist_network,
                candidate_files,
                specialist_image_size,
                tta=not args.no_tta,
                tta_runs=args.tta_runs,
            )
            submission_scores, specialist_count = apply_montreal_specialist(
                submission_probs,
                specialist_logits,
                specialist_indices,
                apply_mask,
                alpha=MONTREAL_SPECIALIST_ALPHA,
            )
            print(f"  spécialiste appliqué sur {specialist_count} images")

            del specialist_network
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            submission_scores = submission_probs

    output_path = write_submission_csv(submission_scores, classes, test_files, Path(args.output))
    print(f"Submission écrite: {output_path} ({len(test_files)} lignes)")


if __name__ == "__main__":
    main()
