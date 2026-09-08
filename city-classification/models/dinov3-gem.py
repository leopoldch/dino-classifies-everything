import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchvision import datasets, transforms
from transformers import AutoModel, AutoConfig
from poutyne import Model, ModelCheckpoint, EarlyStopping, CosineAnnealingLR

from config import Config
from utils import DINOv3GeMClassifier, add_pseudo_labels, split_by_base_image


load_dotenv(find_dotenv())
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = Config()
TRAIN_DIR = config.DATA_DIR / config.COMPETITION / "train"
PSEUDO_CSV = Path(__file__).resolve().parents[1] / "pseudo_labels_confident.csv"
USE_PSEUDO_LABELS = os.getenv("USE_PSEUDO_LABELS") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 392
UNFREEZE_LAST_N = 12
BATCH_SIZE_HEAD = 32
BATCH_SIZE_PARTIAL = 8
BATCH_SIZE_FINAL = 4
EPOCHS_HEAD = 18
EPOCHS_PARTIAL = 18
EPOCHS_FINAL = 12
LR_HEAD = 3.6e-4
LR_CLASSIFIER = 3.2e-4
LR_BACKBONE = 2.2e-6
LR_CLASSIFIER_FINAL = 1.8e-5
LR_BACKBONE_FINAL = 9.0e-7
VAL_SPLIT = 0.1
WEIGHT_DECAY = 0.0047
LABEL_SMOOTHING = 0.05
PATIENCE_HEAD = 5
PATIENCE_PARTIAL = 4
PATIENCE_FINAL = 5
TRAIN_CROP_SCALE_MIN = 0.50
FINAL_CROP_SCALE_MIN = 0.72
JITTER_STRENGTH = 0.18
FINAL_JITTER_STRENGTH = 0.08
AUGMENT_SUFFIXES = ("_flip", "_color", "_gray", "_persp", "_crop", "_rrcrop")
SEEDS = [3, 13, 291]
RANDOM_ERASE_P = 0.04
FINAL_RANDOM_ERASE_P = 0.0
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def resize_for_crop(image_size):
    return round(image_size * 256 / 224)


def build_train_transform(image_size, crop_scale_min, jitter_strength, erase_p, final_phase=False):
    transform_list = [
        transforms.RandomResizedCrop(
            (image_size, image_size),
            scale=(crop_scale_min, 1.0),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.15, p=1.0),
        ], p=0.15 if not final_phase else 0.08),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=4 if final_phase else 6,
                translate=(0.02, 0.02) if final_phase else (0.04, 0.04),
                scale=(0.98, 1.02) if final_phase else (0.94, 1.06),
            ),
        ], p=0.15 if final_phase else 0.25),
        transforms.ColorJitter(
            brightness=jitter_strength,
            contrast=jitter_strength,
            saturation=jitter_strength,
        ),
        transforms.RandomAutocontrast(p=0.06 if final_phase else 0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        transforms.RandomErasing(p=erase_p, value="random"),
    ]
    return transforms.Compose(transform_list)


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


def make_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )


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
    final_phase=True,
)
val_transform_final = build_eval_transform(FINAL_IMAGE_SIZE)


def train(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    full = datasets.ImageFolder(TRAIN_DIR)
    classes = full.classes
    train_idx, val_idx = split_by_base_image(full.samples, seed, VAL_SPLIT, AUGMENT_SUFFIXES)

    train_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
    val_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)
    train_set_final = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform_final), train_idx)
    val_set_final = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform_final), val_idx)
    train_set_final = add_pseudo_labels(train_set_final, PSEUDO_CSV, classes, train_transform_final, USE_PSEUDO_LABELS)

    train_loader_head = make_loader(train_set, BATCH_SIZE_HEAD, shuffle=True)
    val_loader_head = make_loader(val_set, BATCH_SIZE_HEAD * 2, shuffle=False)
    train_loader_partial = make_loader(train_set, BATCH_SIZE_PARTIAL, shuffle=True)
    val_loader_partial = make_loader(val_set, BATCH_SIZE_PARTIAL * 2, shuffle=False)
    train_loader_final = make_loader(train_set_final, BATCH_SIZE_FINAL, shuffle=True)
    val_loader_final = make_loader(val_set_final, BATCH_SIZE_FINAL * 2, shuffle=False)

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
        len(classes),
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
        callbacks=build_callbacks(f"dinov3-gem-s{seed}-head.pt", PATIENCE_HEAD, EPOCHS_HEAD),
    )
    model.load_weights(f"dinov3-gem-s{seed}-head.pt")

    del model.optimizer
    torch.cuda.empty_cache()

    for param in backbone.parameters():
        param.requires_grad = False
    for block in encoder.layer[-UNFREEZE_LAST_N:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in norm.parameters():
        param.requires_grad = True

    if hasattr(backbone, "gradient_checkpointing_enable"):
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
        callbacks=build_callbacks(f"dinov3-gem-s{seed}-partial.pt", PATIENCE_PARTIAL, EPOCHS_PARTIAL),
    )
    model.load_weights(f"dinov3-gem-s{seed}-partial.pt")

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
        callbacks=build_callbacks(f"dinov3-gem-s{seed}-final.pt", PATIENCE_FINAL, EPOCHS_FINAL),
    )
    model.load_weights(f"dinov3-gem-s{seed}-final.pt")
    print(f"Checkpoint final: dinov3-gem-s{seed}-final.pt")


if __name__ == "__main__":
    for seed in SEEDS:
        print(f"seed {seed}")
        train(seed)
