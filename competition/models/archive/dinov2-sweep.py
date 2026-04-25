import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification
from poutyne import Model, ModelCheckpoint, EarlyStopping, Callback
from config import Config
from utils import DINOv2Wrapper, split_by_base_image

load_dotenv(find_dotenv())
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
wandb.login(key=WANDB_API_KEY)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = Config()
TRAIN_DIR = config.DATA_DIR / config.COMPETITION / "train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT_SUFFIXES = ("_flip", "_color", "_gray", "_persp", "_crop", "_rrcrop")
SEED = 42
VAL_SPLIT = 0.1

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "lr_head":             {"distribution": "log_uniform_values", "min": 1e-4, "max": 2e-3},
        "lr_classifier":       {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-4},
        "lr_backbone":         {"distribution": "log_uniform_values", "min": 5e-7, "max": 2e-5},
        "lr_classifier_final": {"distribution": "log_uniform_values", "min": 1e-5, "max": 2e-4},
        "lr_backbone_final":   {"distribution": "log_uniform_values", "min": 1e-7, "max": 5e-6},
        "weight_decay":        {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
        "unfreeze_last_n":     {"values": [2, 4, 6, 8]},
        "label_smoothing":     {"values": [0.0, 0.05, 0.1, 0.2]},
        "patience_head":       {"values": [3, 4, 5, 6]},
        "patience_partial":    {"values": [3, 4, 5, 6]},
        "crop_scale_min":      {"distribution": "uniform", "min": 0.4, "max": 0.8},
        "jitter_strength":     {"distribution": "uniform", "min": 0.1, "max": 0.5},
    },
}

class WandbLogger(Callback):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs):
        val_acc = logs.get("val_acc", logs.get("val_accuracy", 0.0))
        self.best_val_acc = max(self.best_val_acc, val_acc)
        wandb.log({
            f"{self.phase}/val_accuracy": val_acc,
            f"{self.phase}/val_loss":     logs.get("val_loss", 0.0),
            f"{self.phase}/train_loss":   logs.get("loss", 0.0),
        })


def make_loader(dataset, batch_size, shuffle):
    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    if shuffle:
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def train():
    run = wandb.init()
    c = run.config
    rid = run.id

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    full = datasets.ImageFolder(TRAIN_DIR)
    classes = full.classes
    train_idx, val_idx = split_by_base_image(full.samples, SEED, VAL_SPLIT, AUGMENT_SUFFIXES)

    IMAGE_SIZE, FINAL_SIZE = 224, 336

    js = c.jitter_strength
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(c.crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=js, contrast=js, saturation=js),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_tf_336 = transforms.Compose([
        transforms.RandomResizedCrop((FINAL_SIZE, FINAL_SIZE), scale=(c.crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=js, contrast=js, saturation=js),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf_336 = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(FINAL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    subset = lambda tf, idx: torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=tf), idx)
    train_set     = subset(train_tf,     train_idx)
    val_set       = subset(val_tf,       val_idx)
    train_set_336 = subset(train_tf_336, train_idx)
    val_set_336   = subset(val_tf_336,   val_idx)

    hf_model = AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-with-registers-large", num_labels=len(classes), ignore_mismatched_sizes=True,
    )
    network  = DINOv2Wrapper(hf_model)
    backbone = hf_model.dinov2_with_registers if hasattr(hf_model, "dinov2_with_registers") else hf_model.dinov2

    ckpt_head    = f"sweep_{rid}_head.pt"
    ckpt_partial = f"sweep_{rid}_partial.pt"
    ckpt_final   = f"sweep_{rid}_final.pt"
    
    # tête seulement    
    for param in backbone.parameters():
        param.requires_grad = False

    logger_head = WandbLogger("head")
    model = Model(
        network,
        torch.optim.AdamW(hf_model.classifier.parameters(), lr=c.lr_head, weight_decay=c.weight_decay),
        nn.CrossEntropyLoss(label_smoothing=c.label_smoothing),
        batch_metrics=["accuracy"],
        device=DEVICE,
    )
    model.fit_generator(
        make_loader(train_set, 64, True), make_loader(val_set, 128, False),
        epochs=20,
        callbacks=[ModelCheckpoint(ckpt_head, save_best_only=True), EarlyStopping(patience=c.patience_head), logger_head],
    )
    model.load_weights(ckpt_head)

    # dégél backbone (pas en entier)
    del model.optimizer
    torch.cuda.empty_cache()

    n = c.unfreeze_last_n
    for param in backbone.parameters():
        param.requires_grad = False
    for block in backbone.encoder.layer[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in backbone.layernorm.parameters():
        param.requires_grad = True

    hf_model.gradient_checkpointing_enable()
    backbone_params = list(backbone.layernorm.parameters())
    for block in backbone.encoder.layer[-n:]:
        backbone_params.extend(block.parameters())

    logger_partial = WandbLogger("partial")
    model.optimizer = torch.optim.AdamW(
        [
            {"params": list(hf_model.classifier.parameters()), "lr": c.lr_classifier},
            {"params": backbone_params,                         "lr": c.lr_backbone},
        ],
        weight_decay=c.weight_decay,
    )
    model.fit_generator(
        make_loader(train_set, 16, True), make_loader(val_set, 32, False),
        epochs=20,
        callbacks=[ModelCheckpoint(ckpt_partial, save_best_only=True), EarlyStopping(patience=c.patience_partial), logger_partial],
    )
    model.load_weights(ckpt_partial)

    # fine tuning précis image
    del model.optimizer
    torch.cuda.empty_cache()

    backbone_params = list(backbone.layernorm.parameters())
    for block in backbone.encoder.layer[-n:]:
        backbone_params.extend(block.parameters())

    logger_final = WandbLogger("final")
    model.optimizer = torch.optim.AdamW(
        [
            {"params": list(hf_model.classifier.parameters()), "lr": c.lr_classifier_final},
            {"params": backbone_params,                         "lr": c.lr_backbone_final},
        ],
        weight_decay=c.weight_decay,
    )
    model.fit_generator(
        make_loader(train_set_336, 8, True), make_loader(val_set_336, 16, False),
        epochs=4,
        callbacks=[ModelCheckpoint(ckpt_final, save_best_only=True), EarlyStopping(patience=2), logger_final],
    )

    wandb.run.summary["val_accuracy"] = logger_final.best_val_acc
    run.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="tp2-glo7030", entity="lacha188-universit-laval")
    wandb.agent(sweep_id, train) 