import pathlib
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class BorealTCFusedSample:
    fused_df: pd.DataFrame
    class_name: str
    run_id: str


class BorealTC(Dataset):
    def __init__(self, root: str, transform=None, classes: Optional[list[str]] = None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.columns = [
            "wx", "wy", "wz", "ax", "ay", "az", "curL", "curR", "velL", "velR"
        ]

        class_paths = sorted(
            [d for d in self.root.iterdir() if d.is_dir() and d.stem != "MIXED"]
        )
        self.classes = classes if classes else [d.stem.lower() for d in class_paths]
        self.class_to_idx = {k: i for i, k in enumerate(self.classes)}

        self.samples = []
        for class_path in class_paths:
            class_name = class_path.stem.lower()
            if class_name not in self.classes:
                continue
            for imu_path in sorted(class_path.glob("imu_*.csv")):
                run_id = imu_path.stem.split("_")[1]
                pro_path = class_path / f"pro_{run_id}.csv"

                imu_df = pd.read_csv(imu_path).set_index("time")
                pro_df = pd.read_csv(pro_path).set_index("time")
                imu_df.index = pd.to_timedelta(imu_df.index, unit="s")
                pro_df.index = pd.to_timedelta(pro_df.index, unit="s")

                fused = fuse_measures(imu_df, pro_df, self.columns)
                self.samples.append(BorealTCFusedSample(fused, class_name, run_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> BorealTCFusedSample:
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def fuse_measures(imu_df, pro_df, cols):
    """Aligns IMU and proprioception data on the highest-frequency time index."""
    freq1 = pd.infer_freq(imu_df.index)
    freq2 = pd.infer_freq(pro_df.index)
    highest_freq = min(freq1, freq2) if freq1 and freq2 else freq1 or freq2

    imu_df = imu_df.resample(highest_freq).ffill()
    pro_df = pro_df.resample(highest_freq).ffill()

    aligned = pd.concat([imu_df, pro_df], axis=1).ffill()
    return aligned[cols]


class SlidingWindowDataset(Dataset):
    """Generates sliding windows from the BorealTC fused dataset."""

    def __init__(self, dataset: BorealTC, window_size: int = 170, step_size: int = 50, transform=None):
        self.dataset = dataset
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.windows = self._generate_windows()

    def _generate_windows(self):
        windows = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            total_steps = len(sample.fused_df)
            for start in range(0, total_steps - self.window_size + 1, self.step_size):
                windows.append((idx, start, start + self.window_size))
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start, end = self.windows[idx]
        sample = self.dataset[sample_idx]
        window_df = sample.fused_df.iloc[start:end]
        window_tensor = torch.tensor(window_df.values, dtype=torch.float32)
        result = {
            "window": window_tensor,
            "class_name": sample.class_name,
            "run_id": sample.run_id,
        }
        if self.transform:
            result = self.transform(result)
        return result
