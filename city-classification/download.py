import kagglehub
import shutil
from pathlib import Path
from config import Config

config = Config()

DATASET_NAME = config.COMPETITION
DATA_DIR = config.DATA_DIR

def download():
    DATA_DIR.mkdir(exist_ok=True)

    dest = DATA_DIR / DATASET_NAME
    if dest.exists():
        print(f"{DATASET_NAME} already present in {dest}")
        return

    print(f"download {DATASET_NAME} ...")
    tmp = kagglehub.competition_download(DATASET_NAME)
    shutil.copytree(tmp, dest)
    print(f"done {DATASET_NAME} -> {dest}")

if __name__ == "__main__":
    download()
