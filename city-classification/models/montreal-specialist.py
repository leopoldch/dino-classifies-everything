import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
from dotenv import load_dotenv, find_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchvision import datasets, transforms
from transformers import AutoModel, AutoConfig
from poutyne import Model, ModelCheckpoint, EarlyStopping, CosineAnnealingLR

from config import Config
from utils import DINOv3GeMClassifier, split_by_base_image


load_dotenv(find_dotenv())
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = Config()
TRAIN_DIR = config.DATA_DIR / config.COMPETITION / "train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPECIALIST_CLASSES = ("Montreal", "Quebec", "Boston")
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 392
UNFREEZE_LAST_N = 16
BATCH_SIZE_HEAD = 32
BATCH_SIZE_PARTIAL = 8
BATCH_SIZE_FINAL = 4
EPOCHS_HEAD = 20
EPOCHS_PARTIAL = 18
EPOCHS_FINAL = 14
LR_HEAD = 3.0e-4
LR_CLASSIFIER = 2.5e-4
LR_BACKBONE = 2.0e-6
LR_CLASSIFIER_FINAL = 1.5e-5
LR_BACKBONE_FINAL = 8.0e-7
VAL_SPLIT = 0.1
WEIGHT_DECAY = 0.004727517721490038
LABEL_SMOOTHING = 0.03
PATIENCE_HEAD = 5
PATIENCE_PARTIAL = 4
PATIENCE_FINAL = 5
TRAIN_CROP_SCALE_MIN = 0.50
FINAL_CROP_SCALE_MIN = 0.72
JITTER_STRENGTH = 0.18
FINAL_JITTER_STRENGTH = 0.10
AUGMENT_SUFFIXES = ("_flip", "_color", "_gray", "_persp", "_crop", "_rrcrop")
SEED = 9
RANDOM_ERASE_P = 0.08
FINAL_RANDOM_ERASE_P = 0.0
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class RemappedSubset(Dataset):
    def __init__(self, root_dir, indices, transform, specialist_classes):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.indices = list(indices)
        self.label_mapping = {
            self.dataset.class_to_idx[class_name]: index
            for index, class_name in enumerate(specialist_classes)
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, self.label_mapping[label]


def resize_for_crop(image_size):
    return round(image_size * 256 / 224)


def build_train_transform(image_size, crop_scale_min, jitter_strength, erase_p):
    return transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size), scale=(crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=6,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
            ),
        ], p=0.20),
        transforms.ColorJitter(
            brightness=jitter_strength,
            contrast=jitter_strength,
            saturation=jitter_strength,
        ),
        transforms.RandomAutocontrast(p=0.08),
        transforms.RandomApply([
            transforms.RandomAdjustSharpness(sharpness_factor=1.4),
        ], p=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        transforms.RandomErasing(p=erase_p, value="random"),
    ])


def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize(resize_for_crop(image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def build_callbacks(checkpoint_name, patience, epochs):
    return [
        ModelCheckpoint(checkpoint_name, save_best_only=True),
        EarlyStopping(patience=patience),
        CosineAnnealingLR(T_max=epochs, eta_min=0.0),
    ]


def build_specialist_indices(full_dataset):
    selected_indices = [
        index
        for index, (_, label) in enumerate(full_dataset.samples)
        if full_dataset.classes[label] in SPECIALIST_CLASSES
    ]
    selected_samples = [full_dataset.samples[index] for index in selected_indices]
    train_pos, val_pos = split_by_base_image(selected_samples, SEED, VAL_SPLIT, AUGMENT_SUFFIXES)
    train_indices = [selected_indices[index] for index in train_pos]
    val_indices = [selected_indices[index] for index in val_pos]
    return train_indices, val_indices


train_transform = build_train_transform(
    IMAGE_SIZE,
    TRAIN_CROP_SCALE_MIN,
    JITTER_STRENGTH,
    RANDOM_ERASE_P,
)
val_transform = build_eval_transform(IMAGE_SIZE)
train_transform_final = build_train_transform(
    FINAL_IMAGE_SIZE,
    FINAL_CROP_SCALE_MIN,
    FINAL_JITTER_STRENGTH,
    FINAL_RANDOM_ERASE_P,
)
val_transform_final = build_eval_transform(FINAL_IMAGE_SIZE)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    full = datasets.ImageFolder(TRAIN_DIR)
    missing = [class_name for class_name in SPECIALIST_CLASSES if class_name not in full.class_to_idx]
    if missing:
        raise SystemExit(f"Missing specialist classes: {missing}")

    train_indices, val_indices = build_specialist_indices(full)
    print(f"Specialist: {SPECIALIST_CLASSES}")
    print(f"Train: {len(train_indices)} images | Val: {len(val_indices)} images")

    train_set = RemappedSubset(TRAIN_DIR, train_indices, train_transform, SPECIALIST_CLASSES)
    val_set = RemappedSubset(TRAIN_DIR, val_indices, val_transform, SPECIALIST_CLASSES)
    train_set_final = RemappedSubset(TRAIN_DIR, train_indices, train_transform_final, SPECIALIST_CLASSES)
    val_set_final = RemappedSubset(TRAIN_DIR, val_indices, val_transform_final, SPECIALIST_CLASSES)

    train_loader_head = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE_HEAD,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader_head = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE_HEAD * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader_partial = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE_PARTIAL,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader_partial = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE_PARTIAL * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader_final = torch.utils.data.DataLoader(
        train_set_final,
        batch_size=BATCH_SIZE_FINAL,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader_final = torch.utils.data.DataLoader(
        val_set_final,
        batch_size=BATCH_SIZE_FINAL * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    kwargs = {"token": HUGGING_FACE_TOKEN} if HUGGING_FACE_TOKEN else {}
    hf_config = AutoConfig.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        **kwargs,
    )
    backbone = AutoModel.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        **kwargs,
    )
    encoder = backbone.model
    norm = backbone.norm

    network = DINOv3GeMClassifier(
        backbone,
        hf_config.hidden_size,
        len(SPECIALIST_CLASSES),
        num_register_tokens=getattr(hf_config, "num_register_tokens", 0),
    )

    for param in backbone.parameters():
        param.requires_grad = False

    model = Model(
        network,
        torch.optim.AdamW(network.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY),
        nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING),
        batch_metrics=["accuracy"],
        device=DEVICE,
    )

    model.fit_generator(
        train_loader_head,
        val_loader_head,
        epochs=EPOCHS_HEAD,
        callbacks=build_callbacks("montreal-specialist-evolved-head.pt", PATIENCE_HEAD, EPOCHS_HEAD),
    )
    model.load_weights("montreal-specialist-evolved-head.pt")

    del model.optimizer
    torch.cuda.empty_cache()

    for param in backbone.parameters():
        param.requires_grad = False
    for block in encoder.layer[-UNFREEZE_LAST_N:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in norm.parameters():
        param.requires_grad = True

    backbone.gradient_checkpointing_enable()

    backbone_params = list(norm.parameters())
    for block in encoder.layer[-UNFREEZE_LAST_N:]:
        backbone_params.extend(block.parameters())

    model.optimizer = torch.optim.AdamW(
        [
            {"params": list(network.classifier.parameters()), "lr": LR_CLASSIFIER},
            {"params": backbone_params, "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    model.fit_generator(
        train_loader_partial,
        val_loader_partial,
        epochs=EPOCHS_PARTIAL,
        callbacks=build_callbacks("montreal-specialist-evolved-partial.pt", PATIENCE_PARTIAL, EPOCHS_PARTIAL),
    )
    model.load_weights("montreal-specialist-evolved-partial.pt")

    del model.optimizer
    torch.cuda.empty_cache()

    backbone_params = list(norm.parameters())
    for block in encoder.layer[-UNFREEZE_LAST_N:]:
        backbone_params.extend(block.parameters())

    model.optimizer = torch.optim.AdamW(
        [
            {"params": list(network.classifier.parameters()), "lr": LR_CLASSIFIER_FINAL},
            {"params": backbone_params, "lr": LR_BACKBONE_FINAL},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    model.fit_generator(
        train_loader_final,
        val_loader_final,
        epochs=EPOCHS_FINAL,
        callbacks=build_callbacks("montreal-specialist-evolved-final.pt", PATIENCE_FINAL, EPOCHS_FINAL),
    )
    model.load_weights("montreal-specialist-evolved-final.pt")
    print("Final specialist checkpoint: montreal-specialist-evolved-final.pt")
