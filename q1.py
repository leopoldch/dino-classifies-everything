import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from pathlib import Path

# Constantes et hyperparamètres
COLORS = ["R","G","B"]
DATA_ROOT   = Path("data/cub200")
BATCH_SIZE  = 32
LR          = 1e-3
NUM_CLASSES = 200
NUM_EPOCHS  = 20
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def compute_mean_std(data_root:str, img_size=224):
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
        # imgs : (B, C, H, W)
        B, C, H, W = imgs.shape
        n          = B * H * W
        mean      += imgs.sum(dim=[0, 2, 3])
        std       += (imgs ** 2).sum(dim=[0, 2, 3])
        n_pixels  += n

    mean /= n_pixels
    std   = (std / n_pixels - mean ** 2).sqrt()

    return mean.tolist(), std.tolist()

def get_dataloaders(data_root:str,mean ,std ,img_size=224,noisy=False):
    train_transform = T.Compose([ 
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(), # pour ajouter un peu d'aléas | a retirer peut etre
        T.ToTensor(), 
        T.Normalize(mean=mean,std=std)
    ])
    test_transform = T.Compose([ 
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean,std=std)
    ])

    train_dataset = torchvision.datasets.ImageFolder(f'{data_root}/train', transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(f'{data_root}/test', transform=test_transform)

    if noisy:
        noisy_train_dataset = NoisyLabelDataset(train_dataset, num_classes=NUM_CLASSES, noise_percentage=0.1)
        train_loader =DataLoader(noisy_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    test_loader   = DataLoader(test_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
   
    return train_loader, test_loader

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
   # parser.add_argument("")    
    # parse args for question a et b 
    # et dernière question bruit

    # aussi pour déterminer les coefficients
    mean, std = compute_mean_std(DATA_ROOT)
    for i in range(len(mean)):
    	print(f"{COLORS[i]} : {mean[i]:.5f} ± {std[i]:.5f}") 
