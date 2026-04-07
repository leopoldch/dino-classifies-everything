from pathlib import Path

class Config:

    def __init__(self):
        self.COMPETITION = "ou-suis-je-h-2026"
        self.DATA_DIR = Path(__file__).resolve().parents[1] / "data"