import csv
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification
from poutyne import Model
from config import Config
from utils import DINOv2Wrapper, TestDataset
import argparse

config = Config()
TEST_DIR = config.DATA_DIR / config.COMPETITION / "test"
TRAIN_DIR = config.DATA_DIR / config.COMPETITION / "train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_test(model, classes, batch_size=BATCH_SIZE):
    test_files = sorted(Path(TEST_DIR).glob("*.jpg"))
    logits = model.predict_dataset(TestDataset(TEST_DIR, val_transform), batch_size=batch_size)
    preds = np.argmax(logits, axis=1)

    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for path, pred in zip(test_files, preds):
            writer.writerow([path.stem, classes[pred]])

    print(f"submission.csv généré ({len(preds)} lignes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Chemin vers le fichier .pt ")
    args = parser.parse_args()

    classes = datasets.ImageFolder(TRAIN_DIR).classes

    dinov2 = AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-large",
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
    )
    network = DINOv2Wrapper(dinov2)
    model = Model(network, device=DEVICE)
    model.load_weights(args.weights)

    make_test(model, classes)
