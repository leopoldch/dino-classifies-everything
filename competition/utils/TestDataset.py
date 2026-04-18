from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, folder_or_files, transform):
        if isinstance(folder_or_files, (str, Path)):
            self.files = sorted(Path(folder_or_files).glob("*.jpg"))
        else:
            self.files = list(folder_or_files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert("RGB"))
