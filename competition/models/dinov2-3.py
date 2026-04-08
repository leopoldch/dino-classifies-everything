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
BATCH_SIZE_HEAD = 64
BATCH_SIZE_TUNE = 24
NUM_WORKERS = 4
VAL_SPLIT = 0.1
LR_HEAD = 5e-4
LR_CLASSIFIER = 1e-4
LR_BACKBONE = 5e-6
WEIGHT_DECAY = 0.01
EPOCHS_HEAD = 20
EPOCHS_TUNE = 12
UNFREEZE_LAST_N_BLOCKS = 4
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


def split_by_base_image(samples):
    groups = {}
    for idx, (path, label) in enumerate(samples):
        stem = Path(path).stem
        for suffix in AUGMENT_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        groups.setdefault((label, stem), []).append(idx)

    by_label = {}
    for key in groups:
        by_label.setdefault(key[0], []).append(key)

    rng = random.Random(SEED)
    train_idx, val_idx = [], []
    for label in sorted(by_label):
        keys = by_label[label]
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * VAL_SPLIT))
        val_keys = set(keys[:n_val])
        for key in keys:
            (val_idx if key in val_keys else train_idx).extend(groups[key])

    return sorted(train_idx), sorted(val_idx)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    full = datasets.ImageFolder(TRAIN_DIR)
    classes = full.classes

    train_idx, val_idx = split_by_base_image(full.samples)

    train_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
    val_set = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)

    train_loader_head = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE_HEAD, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader_head = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE_HEAD * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    train_loader_tune = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE_TUNE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader_tune = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE_TUNE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    dinov2 = AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-large",
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
    )
    network = DINOv2Wrapper(dinov2)

    #tête seulement
    for param in dinov2.dinov2.parameters():
        param.requires_grad = False

    model = Model(
        network,
        torch.optim.AdamW(dinov2.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY),
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

    #derniers N blocs dégelés
    for block in dinov2.dinov2.encoder.layer[-UNFREEZE_LAST_N_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in dinov2.dinov2.layernorm.parameters():
        param.requires_grad = True

    backbone_params = list(dinov2.dinov2.layernorm.parameters())
    for block in dinov2.dinov2.encoder.layer[-UNFREEZE_LAST_N_BLOCKS:]:
        backbone_params.extend(block.parameters())

    model.optimizer = torch.optim.AdamW(
        [
            {"params": list(dinov2.classifier.parameters()), "lr": LR_CLASSIFIER},
            {"params": backbone_params, "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    model.fit_generator(
        train_loader_tune, val_loader_tune,
        epochs=EPOCHS_TUNE,
        callbacks=[
            ModelCheckpoint("dinov2-3.pt", save_best_only=True),
            EarlyStopping(patience=4),
        ],
    )

    model.load_weights("dinov2-3.pt")
    make_test(model, classes)
