from pathlib import Path
from PIL import Image
from torchvision import transforms
from config import Config

# Offline data augmentation (saves augmented copies to disk).
# Not used in the final pipeline — augmentation is applied on-the-fly
# in the training scripts instead.


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

if __name__ == "__main__":
    apply_data_augmentation()
