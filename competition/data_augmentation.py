from pathlib import Path
from PIL import Image
from torchvision import transforms
from config import Config

# PAS UTILISÉ !!! 
# SIMPLE IDÉE !

# Normalement pas la bonne façon de faire 
# fait directement dans les modèles 
# cf : 
"""
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""


TRAIN_DIR = Config().DATA_DIR / Config().COMPETITION / "train"
SUFFIXES  = ("_flip", "_color", "_gray", "_persp", "_crop", "_rrcrop")

color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
perspective  = transforms.RandomPerspective(distortion_scale=0.3, p=1.0)


def apply_flip(img, img_path):
    img.transpose(Image.FLIP_LEFT_RIGHT).save(img_path.with_stem(img_path.stem + "_flip"))

def apply_color_jitter(img, img_path):
    color_jitter(img).save(img_path.with_stem(img_path.stem + "_color"))

def apply_grayscale(img, img_path):
    img.convert("L").convert("RGB").save(img_path.with_stem(img_path.stem + "_gray"))

def apply_perspective(img, img_path):
    perspective(img).save(img_path.with_stem(img_path.stem + "_persp"))

def apply_random_crop(img, img_path):
    w, h = img.size
    transforms.RandomCrop((int(h * 0.85), int(w * 0.85)))(img).save(img_path.with_stem(img_path.stem + "_crop"))

def apply_random_resized_crop(img, img_path):
    w, h = img.size
    transforms.RandomResizedCrop((h, w), scale=(0.6, 1.0))(img).save(img_path.with_stem(img_path.stem + "_rrcrop"))


def apply_data_augmentation():
    for img_path in sorted(TRAIN_DIR.rglob("*.jpg")):
        if any(img_path.stem.endswith(s) for s in SUFFIXES):
            continue
        img = Image.open(img_path).convert("RGB")
        apply_flip(img, img_path)
        apply_color_jitter(img, img_path)
        apply_grayscale(img, img_path)
        apply_perspective(img, img_path)
        apply_random_crop(img, img_path)
        apply_random_resized_crop(img, img_path)

# total de données ajoutées:
# 135 000 + 22 500 

if __name__ == "__main__":
    apply_data_augmentation()
