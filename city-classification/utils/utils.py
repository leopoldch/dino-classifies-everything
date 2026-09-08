import os
import random
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

from config import Config
from .DINOv2Classifier import DINOv2Classifier
from .DINOv2Wrapper import DINOv2Wrapper
from .DINOv3Classifier import DINOv3Classifier
from .DINOv3GeMClassifier import DINOv3GeMClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT = 0.1
SEED = 9
AUGMENT_SUFFIXES = ("_flip", "_color", "_gray", "_persp", "_crop", "_rrcrop")
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
DEFAULT_TTA_RUNS = 8
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


class PseudoLabelDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, transform):
        self.samples = []
        self.transform = transform

        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                class_name = row["class"]
                if class_name not in class_to_idx:
                    continue
                self.samples.append((Path(row["image_path"]), class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def add_pseudo_labels(train_dataset, pseudo_csv_path, classes, transform, enabled=False):
    if not enabled:
        return train_dataset

    pseudo_csv_path = Path(pseudo_csv_path)
    if not pseudo_csv_path.exists():
        print(f"Pseudo-labels skipped: file not found ({pseudo_csv_path})")
        return train_dataset

    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    pseudo_dataset = PseudoLabelDataset(pseudo_csv_path, class_to_idx, transform)
    if len(pseudo_dataset) == 0:
        print(f"Pseudo-labels skipped: no valid examples in {pseudo_csv_path}")
        return train_dataset

    print(f"Pseudo-labels added to training set: {len(pseudo_dataset)} images from {pseudo_csv_path}")
    return ConcatDataset([train_dataset, pseudo_dataset])


def split_by_base_image(samples, seed, val_split, augment_suffixes):
    groups = {}
    for idx, (path, label) in enumerate(samples):
        stem = Path(path).stem
        for suffix in augment_suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        groups.setdefault((label, stem), []).append(idx)

    by_label = {}
    for key in groups:
        by_label.setdefault(key[0], []).append(key)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in sorted(by_label):
        keys = by_label[label]
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_split))
        val_keys = set(keys[:n_val])
        for key in keys:
            (val_idx if key in val_keys else train_idx).extend(groups[key])

    return sorted(train_idx), sorted(val_idx)


def resize_for_crop(image_size):
    return round(image_size * 256 / 224)


def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize(resize_for_crop(image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def build_tta_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            (image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def build_flip_transform(image_size):
    return transforms.Compose([
        transforms.Resize(resize_for_crop(image_size)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_prediction_transforms(image_size, tta=False, tta_runs=DEFAULT_TTA_RUNS):
    transforms_to_run = [build_eval_transform(image_size)]
    if tta:
        transforms_to_run.append(build_flip_transform(image_size))
        for _ in range(tta_runs):
            transforms_to_run.append(build_tta_transform(image_size))
    return transforms_to_run


def find_weight_files(root_dir, paths=None, include_legacy=False):
    if paths:
        return [Path(path) for path in paths]

    weights_dir = Path(root_dir) / "weights"
    candidates = sorted(weights_dir.glob("*.pt")) + sorted(weights_dir.glob("*.pth"))
    if include_legacy:
        candidates += sorted(weights_dir.glob("*.p"))
    return candidates


def load_state_dict(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            return state["model_state_dict"]
    return state


def detect_model_kind(state_dict):
    keys = list(state_dict.keys())
    if any(key.startswith("backbone.encoder.layer.") or key.startswith("backbone.layernorm.") for key in keys):
        return "dinov2-with-registers-large-custom"
    if "gem_p" in keys and any(key.startswith("backbone.model.") or key.startswith("backbone.norm.") for key in keys):
        return "dinov3-gem"
    if any(key.startswith("backbone.model.") or key.startswith("backbone.norm.") for key in keys):
        return "dinov3"
    if any("dinov2_with_registers" in key for key in keys):
        return "dinov2-with-registers-large"
    return "dinov2-large"


def infer_image_size(weights_path, image_size=None):
    if image_size is not None:
        return image_size
    stem = Path(weights_path).stem.lower()
    if "final" not in stem:
        return 224
    if "evolved" in stem or "gem" in stem:
        return 392
    return 336


def build_network(kind, num_classes):
    token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")

    if kind in {"dinov3", "dinov3-gem"}:
        model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        kwargs = {"token": token} if token else {}
        hf_config = AutoConfig.from_pretrained(model_id, **kwargs)
        backbone = AutoModel.from_pretrained(model_id, **kwargs)
        classifier_cls = DINOv3GeMClassifier if kind == "dinov3-gem" else DINOv3Classifier
        return classifier_cls(
            backbone,
            hf_config.hidden_size,
            num_classes,
            num_register_tokens=getattr(hf_config, "num_register_tokens", 0),
        )

    if kind == "dinov2-with-registers-large-custom":
        model_id = "facebook/dinov2-with-registers-large"
        kwargs = {"token": token} if token else {}
        hf_config = AutoConfig.from_pretrained(model_id, **kwargs)
        backbone = AutoModel.from_pretrained(model_id, **kwargs)
        return DINOv2Classifier(
            backbone,
            hf_config.hidden_size,
            num_classes,
            num_register_tokens=getattr(hf_config, "num_register_tokens", 0),
        )

    model_id = f"facebook/{kind}"
    hf_model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return DINOv2Wrapper(hf_model)


def load_network_from_weights(weights_path, num_classes):
    state_dict = load_state_dict(weights_path)
    kind = detect_model_kind(state_dict)
    network = build_network(kind, num_classes)
    network.load_state_dict(state_dict, strict=True)
    return network, kind, state_dict


def find_test_images(test_dir):
    test_dir = Path(test_dir)
    files = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(test_dir.glob(pattern))
    return sorted(files)


def write_submission_csv(logits, classes, test_files, output_path):
    preds = logits.argmax(axis=1)
    output_path = Path(output_path)

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "class"])
        for path, pred in zip(test_files, preds):
            writer.writerow([path.stem, classes[pred]])

    return output_path


def build_validation_dataset(image_size, batch_size):
    config = Config()
    train_dir = config.DATA_DIR / config.COMPETITION / "train"
    full_dataset = datasets.ImageFolder(train_dir)
    _, val_indices = split_by_base_image(full_dataset.samples, SEED, VAL_SPLIT, AUGMENT_SUFFIXES)
    val_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=build_eval_transform(image_size)),
        val_indices,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    return val_loader, full_dataset.classes, full_dataset, val_indices
