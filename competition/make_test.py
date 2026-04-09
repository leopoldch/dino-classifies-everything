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
DEFAULT_MODEL_NAME = "facebook/dinov2-large"


def resize_for_crop(image_size):
    return round(image_size * 256 / 224)


def build_eval_transform(image_size, hflip=False):
    ops = [
        transforms.Resize(resize_for_crop(image_size)),
        transforms.CenterCrop(image_size),
    ]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


def predict_test_logits(model, batch_size=BATCH_SIZE, image_size=224, tta=False):
    transforms_to_run = [build_eval_transform(image_size)]
    if tta:
        transforms_to_run.append(build_eval_transform(image_size, hflip=True))

    logits = [
        model.predict_dataset(TestDataset(TEST_DIR, transform), batch_size=batch_size)
        for transform in transforms_to_run
    ]
    return np.mean(logits, axis=0)


def write_submission(logits, classes, output_path):
    test_files = sorted(Path(TEST_DIR).glob("*.jpg"))
    preds = np.argmax(logits, axis=1)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "class"])
        for path, pred in zip(test_files, preds):
            writer.writerow([path.stem, classes[pred]])

    print(f"{output_path} généré ({len(preds)} lignes)")


def make_test(model, classes, batch_size=BATCH_SIZE, image_size=224, output_path="submission.csv"):
    logits = predict_test_logits(model, batch_size=batch_size, image_size=image_size, tta=False)
    write_submission(logits, classes, output_path)


def make_test_tta(model, classes, batch_size=BATCH_SIZE, image_size=224, output_path="submission.csv"):
    logits = predict_test_logits(model, batch_size=batch_size, image_size=image_size, tta=True)
    write_submission(logits, classes, output_path)


def normalize_values(weights, values, default_value, option_name):
    if values is None:
        return [default_value] * len(weights)
    if len(values) == 1 and len(weights) > 1:
        return values * len(weights)
    if len(values) != len(weights):
        raise ValueError(f"Fournir soit un seul {option_name}, soit une valeur par poids.")
    return values


def load_model(weights_path, model_name, num_labels):
    hf_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    network = DINOv2Wrapper(hf_model)
    model = Model(network, device=DEVICE)
    model.load_weights(weights_path)
    return model


def make_test_ensemble(
    weight_paths,
    classes,
    model_names=None,
    image_sizes=None,
    batch_size=BATCH_SIZE,
    tta=True,
    output_path="submission.csv",
):
    model_names = normalize_values(weight_paths, model_names, DEFAULT_MODEL_NAME, "--model-name")
    image_sizes = normalize_values(weight_paths, image_sizes, 224, "--image-size")
    avg_logits = None

    for weights_path, model_name, image_size in zip(weight_paths, model_names, image_sizes):
        model = load_model(weights_path, model_name, len(classes))
        logits = predict_test_logits(model, batch_size=batch_size, image_size=image_size, tta=tta)
        avg_logits = logits if avg_logits is None else avg_logits + logits
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_logits /= len(weight_paths)
    write_submission(avg_logits, classes, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="+", help="Un ou plusieurs fichiers .pt")
    parser.add_argument("--model-name", nargs="*", dest="model_names")
    parser.add_argument("--image-size", nargs="*", type=int, dest="image_sizes")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    classes = datasets.ImageFolder(TRAIN_DIR).classes
    make_test_ensemble(
        args.weights,
        classes,
        model_names=args.model_names,
        image_sizes=args.image_sizes,
        tta=not args.no_tta,
        output_path=args.output,
    )
