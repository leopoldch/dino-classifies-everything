import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoModel
from question_1 import separate_train_test
from deeplib.training import train


class NoisyLabelDataset(Dataset):
    def __init__(self, original_dataset: Dataset, num_classes: int, noise_percentage: float = 0.1):
        """
        Modifies the labels of a dataset to introduce noise.
        :param original_dataset: Original dataset to wrap.
        :param num_classes: Number of classes in the dataset.
        :param noise_percentage: Fraction of labels to introduce noise (e.g., 0.05 for 5% noise).
        """
        self.original_dataset = original_dataset
        self.noise_percentage = noise_percentage
        self.num_samples = len(original_dataset)
        self.num_classes = num_classes
        num_errors = int(noise_percentage * self.num_samples)
        self.error_indices = random.sample(range(self.num_samples), num_errors)
        self.errors = np.random.randint(0, self.num_classes, (num_errors,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        data, label = self.original_dataset[index]
        if index in self.error_indices:
            label = self.errors[self.error_indices.index(index)]

        return data, label

COLORS = ["R","G","B"]
DATA_ROOT   = Path("data/cub200")
BATCH_SIZE  = 32
LR          = 1e-3
NUM_CLASSES = 200
EPOCHS  = 20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_mean_std(data_root:str=DATA_ROOT, img_size=224):
    """
    Calcule la moyenne et l'écart-type canal par canal
    sur le split d'entraînement de CUB-200-2011.
    """
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(), 
    ])
    dataset = torchvision.datasets.ImageFolder(f'{data_root}/train', transform=tf)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n_pixels = 0

    for imgs, _ in loader:
        B, C, H, W = imgs.shape
        n          = B * H * W
        mean      += imgs.sum(dim=[0, 2, 3])
        std       += (imgs ** 2).sum(dim=[0, 2, 3])
        n_pixels  += n

    mean /= n_pixels
    std   = (std / n_pixels - mean ** 2).sqrt()

    return mean.tolist(), std.tolist()


def ensure_cub200_split(data_root: str | Path = DATA_ROOT):
    data_root = Path(data_root)
    train_root = data_root / "train"
    test_root = data_root / "test"

    if train_root.exists() and test_root.exists():
        return

    raw_root = Path("data/CUB_200_2011/CUB_200_2011/images")
    if not raw_root.exists():
        raise FileNotFoundError(
            f"dataset non trouvé dans {data_root} "
        )

    separate_train_test(raw_root, train_root, test_root)


def get_datasets(mean ,std ,img_size=224,data_root:str=DATA_ROOT,noisy=False):
    ensure_cub200_split(data_root)
    train_transform = T.Compose([ 
        T.Resize((img_size,img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(), 
        T.Normalize(mean=mean,std=std)
    ])
    test_transform = T.Compose([
        T.Resize((img_size,img_size)), 
        T.ToTensor(),
        T.Normalize(mean=mean,std=std)
    ])

    train_dataset = torchvision.datasets.ImageFolder(f'{data_root}/train', transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(f'{data_root}/test', transform=test_transform)

    if noisy:
        train_dataset = NoisyLabelDataset(train_dataset, num_classes=NUM_CLASSES, noise_percentage=0.1)
    return train_dataset, test_dataset


# 1
def build_resnet18_random() -> nn.Module:
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
# 2
def build_resnet18_freeze_all_conv() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
 
    # Geler tous les paramètres existants
    for param in model.parameters():
        param.requires_grad = False
 
    # Nouvelle tête (requires_grad=True par défaut)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
# 3
def build_resnet18_freeze_layer1() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
 
    # Geler conv1, bn1 et layer1 comme demandé
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
 
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
# 4
def build_resnet18_finetune_all() -> nn.Module:
    model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# 5
class DINOv2Classifier(nn.Module):
    """
    Backbone DINOv2-small (HuggingFace) entièrement gelé.
    Seule la couche linéaire 'classifier' est entraînable.
    Le token [CLS] (position 0 de last_hidden_state) sert de représentation.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
 
        # Geler tout le backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
 
        # Tête linéaire – hidden_size = 384 pour DINOv2-small
        hidden_size      = self.backbone.config.hidden_size
        self.classifier  = nn.Linear(hidden_size, num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
        return self.classifier(cls_token)
 
 
def build_dinov2_small() -> nn.Module:
    return DINOv2Classifier(num_classes=NUM_CLASSES)

#### Fonction pour récupérer les accuracys 
def run_training(model: nn.Module, train_dataset:Dataset,
                 test_dataset:Dataset, config_name: str,
                 num_epochs: int = EPOCHS, lr: float = LR):

    optimizer = optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    history = train(
        network=model,
        optimizer=optimizer,
        dataset=train_dataset,
        n_epoch=num_epochs,
        batch_size=BATCH_SIZE,
        criterion=criterion
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device(DEVICE)
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            correct += model(inputs).argmax(1).eq(labels).sum().item()
            total   += labels.size(0)
    test_acc = (correct / total) *100
    return history, test_acc

def main(mean, std, noisy=False):
    set_seed()
    train_dt, test_dt = get_datasets(mean,std,img_size=224,noisy=noisy)
    configs = [
        ("1 - ResNet18 aléatoire",          build_resnet18_random),
        ("2 - ResNet18 PE, conv gelés",     build_resnet18_freeze_all_conv),
        ("3 - ResNet18 PE, layer1 gelé",    build_resnet18_freeze_layer1),
        ("4 - ResNet18 PE, fine-tune all",  build_resnet18_finetune_all),
        ("5 - DINOv2 Small, backbone gelé", build_dinov2_small),
    ]

    results = {}
    for name, builder in configs:
        print(f"Training and testing : {name}")
        model = builder()
        history, test_acc = run_training(
            model, train_dt, test_dt, config_name=name
        )
        results[name] = {"history": history, "test_acc": test_acc}

    return results


def print_results(results):
    for name, data in results.items():
        history = data['history'].history
        final_train_acc = history['acc'][-1]
        test_acc = data['test_acc']
    
        print(f"--- Configuration : {name} ---")
        print(f"   Accuracy Entraînement : {final_train_acc:.2f}%")
        print(f"   Accuracy Test          : {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q","--question",choices=["a","b","d"],default="a")   
    parser.add_argument("--coeff",action="store_true")
    args = parser.parse_args()

    if args.coeff:
        cub_200_mean, cub_200_std = compute_mean_std()
        for i in range(len(cub_200_mean)):
            print(f"{COLORS[i]} : {cub_200_mean[i]:.3f} ± {cub_200_std[i]:.3f}")
        exit(0)

    if args.question == "a":
        results = main(IMAGENET_MEAN, IMAGENET_STD)
        print_results(results)
    elif args.question == "b":
        cub_200_mean, cub_200_std = compute_mean_std()
        for i in range(len(cub_200_mean)):
            print(f"{COLORS[i]} : {cub_200_mean[i]:.3f} ± {cub_200_std[i]:.3f}")
        results = main(cub_200_mean, cub_200_std)
        print_results(results)
    elif args.question == "d":
        cub_200_mean, cub_200_std = compute_mean_std()
        for i in range(len(cub_200_mean)):
            print(f"{COLORS[i]} : {cub_200_mean[i]:.3f} ± {cub_200_std[i]:.3f}")
        results = main(cub_200_mean, cub_200_std,noisy=True)
        print_results(results)
