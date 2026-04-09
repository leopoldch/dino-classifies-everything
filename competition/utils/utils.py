import random
from pathlib import Path


def split_by_base_image(samples, seed, val_split, augment_suffixes):
    groups = {}
    for idx, (path, label) in enumerate(samples):
        stem = Path(path).stem
        for suffix in augment_suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        groups.setdefault((label, stem), []).append(idx)

    by_label = {}
    for key in groups:
        by_label.setdefault(key[0], []).append(key)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in sorted(by_label):
        keys = by_label[label]
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_split))
        val_keys = set(keys[:n_val])
        for key in keys:
            (val_idx if key in val_keys else train_idx).extend(groups[key])

    return sorted(train_idx), sorted(val_idx)
