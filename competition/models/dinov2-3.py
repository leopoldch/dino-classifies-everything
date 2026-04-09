import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification
from poutyne import Model, ModelCheckpoint, EarlyStopping
from config import Config
from make_test import make_test_tta
from utils import DINOv2Wrapper, split_by_base_image

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = Config()
TRAIN_DIR = config.DATA_DIR / config.COMPETITION / "train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 336
UNFREEZE_LAST_N = 4
BATCH_SIZE_HEAD = 64
BATCH_SIZE_PARTIAL = 16
BATCH_SIZE_FINAL = 8
EPOCHS_HEAD = 20
EPOCHS_PARTIAL = 20
EPOCHS_FINAL = 4
LR_HEAD = 5e-4
LR_CLASSIFIER = 1e-4
LR_BACKBONE = 5e-6
LR_CLASSIFIER_FINAL = 5e-5
LR_BACKBONE_FINAL = 2e-6
VAL_SPLIT = 0.1
WEIGHT_DECAY = 0.01
AUGMENT_SUFFIXES = ("_flip", "_color", "_gray", "_persp", "_crop", "_rrcrop")
SEED = 42

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform_336 = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(FINAL_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform_336 = transforms.Compose([
    transforms.RandomResizedCrop((FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE), scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    full = datasets.ImageFolder(TRAIN_DIR)
    classes = full.classes
    train_idx, val_idx = split_by_base_image(full.samples, SEED, VAL_SPLIT, AUGMENT_SUFFIXES)

    train_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
    val_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)
    train_set_336 = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform_336), train_idx)
    val_set_336 = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform_336), val_idx)

    train_loader_head = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE_HEAD, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader_head = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE_HEAD * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    train_loader_partial = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE_PARTIAL, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader_partial = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE_PARTIAL * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    train_loader_final = torch.utils.data.DataLoader(
        train_set_336, batch_size=BATCH_SIZE_FINAL, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader_final = torch.utils.data.DataLoader(
        val_set_336, batch_size=BATCH_SIZE_FINAL * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    hf_model = AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-with-registers-large", num_labels=len(classes), ignore_mismatched_sizes=True,
    )
    network = DINOv2Wrapper(hf_model)
    backbone = hf_model.dinov2_with_registers if hasattr(hf_model, "dinov2_with_registers") else hf_model.dinov2

    # tête seulement
    for param in backbone.parameters():
        param.requires_grad = False

    model = Model(
        network,
        torch.optim.AdamW(hf_model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY),
        nn.CrossEntropyLoss(),
        batch_metrics=["accuracy"],
        device=DEVICE,
    )

    model.fit_generator(
        train_loader_head, val_loader_head,
        epochs=EPOCHS_HEAD,
        callbacks=[
            ModelCheckpoint("dinov2-3-head.pt", save_best_only=True),
            EarlyStopping(patience=4),
        ],
    )
    model.load_weights("dinov2-3-head.pt")

    # dégel partiel avec LR différentiel
    del model.optimizer
    torch.cuda.empty_cache()

    for param in backbone.parameters():
        param.requires_grad = False
    for block in backbone.encoder.layer[-UNFREEZE_LAST_N:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in backbone.layernorm.parameters():
        param.requires_grad = True

    hf_model.gradient_checkpointing_enable()

    backbone_params = list(backbone.layernorm.parameters())
    for block in backbone.encoder.layer[-UNFREEZE_LAST_N:]:
        backbone_params.extend(block.parameters())

    model.optimizer = torch.optim.AdamW(
        [
            {"params": list(hf_model.classifier.parameters()), "lr": LR_CLASSIFIER},
            {"params": backbone_params, "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    model.fit_generator(
        train_loader_partial, val_loader_partial,
        epochs=EPOCHS_PARTIAL,
        callbacks=[
            ModelCheckpoint("dinov2-3-partial.pt", save_best_only=True),
            EarlyStopping(patience=4),
        ],
    )
    model.load_weights("dinov2-3-partial.pt")

    # fine-tuning à 336px avec LR encore plus bas
    del model.optimizer
    torch.cuda.empty_cache()

    backbone_params = list(backbone.layernorm.parameters())
    for block in backbone.encoder.layer[-UNFREEZE_LAST_N:]:
        backbone_params.extend(block.parameters())

    model.optimizer = torch.optim.AdamW(
        [
            {"params": list(hf_model.classifier.parameters()), "lr": LR_CLASSIFIER_FINAL},
            {"params": backbone_params, "lr": LR_BACKBONE_FINAL},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    model.fit_generator(
        train_loader_final, val_loader_final,
        epochs=EPOCHS_FINAL,
        callbacks=[
            ModelCheckpoint("dinov2-3-final.pt", save_best_only=True),
            EarlyStopping(patience=2),
        ],
    )
    model.load_weights("dinov2-3-final.pt")
    make_test_tta(model, classes, image_size=FINAL_IMAGE_SIZE)
