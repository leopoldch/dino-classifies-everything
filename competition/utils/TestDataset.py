from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, folder, transform):
        self.files = sorted(Path(folder).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert("RGB"))
