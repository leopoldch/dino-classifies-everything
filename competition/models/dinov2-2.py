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
from utils import DINOv2Wrapper, split_by_base_image
from make_test import make_test

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = Config()
TRAIN_DIR = config.DATA_DIR / config.COMPETITION / "train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_HEAD = 64
BATCH_SIZE_FULL = 16
NUM_WORKERS = 4
VAL_SPLIT = 0.1
LR_HEAD = 5e-4
LR_FULL = 1e-5
EPOCHS_HEAD = 20
EPOCHS_FULL = 15
AUGMENT_SUFFIXES = ("_flip", "_color", "_gray", "_persp", "_crop", "_rrcrop")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SEED = 42

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    full = datasets.ImageFolder(TRAIN_DIR)
    classes = full.classes

    train_idx, val_idx = split_by_base_image(full.samples, SEED, VAL_SPLIT, AUGMENT_SUFFIXES)

    train_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
    val_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)

    train_loader_head = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE_HEAD,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader_head = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE_HEAD * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader_full = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE_FULL,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader_full = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE_FULL * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    dinov2 = AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-large",
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
    )
    network = DINOv2Wrapper(dinov2)

    for param in dinov2.dinov2.parameters():
        param.requires_grad = False

    model = Model(
        network,
        torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=LR_HEAD),
        nn.CrossEntropyLoss(),
        batch_metrics=["accuracy"],
        device=DEVICE,
    )

    model.fit_generator(
        train_loader_head, val_loader_head,
        epochs=EPOCHS_HEAD,
        callbacks=[
            EarlyStopping(patience=5)
            ],
    )

    del model.optimizer
    torch.cuda.empty_cache()
    dinov2.gradient_checkpointing_enable()

    for param in dinov2.dinov2.parameters():
        param.requires_grad = True

    model.optimizer = torch.optim.AdamW(network.parameters(), lr=LR_FULL, weight_decay=0.01)

    model.fit_generator(
        train_loader_full, val_loader_full,
        epochs=EPOCHS_FULL,
        callbacks=[
            ModelCheckpoint("dinov2-2.pt", save_best_only=True),
            EarlyStopping(patience=5),
        ],
    )

    model.load_weights("dinov2-2.pt")
    make_test(model, classes)
