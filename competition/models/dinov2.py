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
from utils import DINOv2Wrapper
from make_test import make_test

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = Config()
TRAIN_DIR = config.DATA_DIR / config.COMPETITION / "train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 5e-4
VAL_SPLIT = 0.1
SEED = 42
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

if __name__ == "__main__":
    full = datasets.ImageFolder(TRAIN_DIR)
    classes = full.classes

    original_idx = [i for i, (path, _) in enumerate(full.samples)
                    if not any(Path(path).stem.endswith(s) for s in AUGMENT_SUFFIXES)]
    random.shuffle(original_idx)
    n_val = int(len(original_idx) * VAL_SPLIT)
    val_idx = set(original_idx[:n_val])
    train_idx = [i for i in range(len(full.samples)) if i not in val_idx]

    train_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
    val_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), list(val_idx))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE * 2,
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
    for param in dinov2.dinov2.parameters():
        param.requires_grad = False

    network = DINOv2Wrapper(dinov2)

    model = Model(
        network,
        torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=LR),
        nn.CrossEntropyLoss(),
        batch_metrics=["accuracy"],
        device=DEVICE,
    )

    model.fit_generator(
        train_loader, val_loader,
        epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint("dinov2.pt", save_best_only=True),
            EarlyStopping(patience=5),
        ],
    )

    model.load_weights("dinov2.pt")
    make_test(model, classes)
