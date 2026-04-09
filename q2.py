import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.utils.data import Subset, DataLoader
from positional_encodings.torch_encodings import PositionalEncoding1D
from poutyne import Model
from borealtc import BorealTC, SlidingWindowDataset
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEARCH_EPOCHS = 10
FINAL_EPOCHS = 30
BATCH_SIZE = 64
WINDOW_SIZE = 170
STEP_SIZE = 10

WANDB_PROJECT = "tp2-glo7030-q2"
WANDB_ENTITY = "lacha188-universit-laval"

RNN_RANDOM_CONFIGS = 20
LSTM_RANDOM_CONFIGS = 20
TRANSFORMER_RANDOM_CONFIGS = 40


class RNNClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, num_classes=5, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, num_classes=5, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=5,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding1D(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.proj(x)
        x = x + self.pos(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.encoder(x)
        return self.fc(x[:, 0, :])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# fonction donnée dans le TP
def split_train_test(dataset):
    classes = dataset.classes
    test_indices = []
    for c in classes:
        indices = [i for i, d in enumerate(dataset) if d.class_name == c]
        test_indices.append(random.choice(indices))
    train_indices = list(set(range(len(dataset))) - set(test_indices))
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def get_dataloaders(batch_size=BATCH_SIZE, val_ratio=0.15, seed=42):
    set_seed(seed)

    dataset = BorealTC("data/borealtc")
    class_to_idx = dataset.class_to_idx
    num_classes = len(dataset.classes)

    full_train_dataset, test_dataset = split_train_test(dataset)

    n = len(full_train_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    n_val = int(n * val_ratio)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_dataset = SlidingWindowDataset(train_dataset, WINDOW_SIZE, STEP_SIZE)
    val_dataset = SlidingWindowDataset(val_dataset, WINDOW_SIZE, STEP_SIZE)
    test_dataset = SlidingWindowDataset(test_dataset, WINDOW_SIZE, STEP_SIZE)

    def collate_fn(batch):
        windows = torch.stack([sample["window"] for sample in batch])
        labels = torch.tensor([class_to_idx[sample["class_name"]] for sample in batch])
        return windows, labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, num_classes


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def get_acc_keys(history):
    if "acc" in history[0] and "val_acc" in history[0]:
        return "acc", "val_acc"
    return "accuracy", "val_accuracy"


def config_to_name(config):
    parts = []
    for key, value in config.items():
        parts.append(f"{key}-{value}")
    return "_".join(parts)


def sample_rnn_lstm_configs(n, seed=42):
    rng = random.Random(seed)

    hidden_sizes = [32, 64, 128, 256]
    num_layers_list = [2, 3, 4, 5]
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]

    configs = []
    seen = set()

    while len(configs) < n:
        config = {
            "hidden_size": rng.choice(hidden_sizes),
            "num_layers": rng.choice(num_layers_list),
            "dropout": rng.choice(dropouts),
        }

        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)

    return configs


def sample_transformer_configs(n, seed=42):
    rng = random.Random(seed)

    d_models = [32, 64, 128, 256]
    nheads = [2, 4, 8]
    num_layers_list = [2, 3, 4, 5]
    dim_feedforwards = [64, 128, 256, 512]
    dropouts = [0.1, 0.2, 0.3, 0.4]

    configs = []
    seen = set()

    while len(configs) < n:
        d_model = rng.choice(d_models)
        nhead = rng.choice(nheads)

        if d_model % nhead != 0:
            continue

        config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": rng.choice(num_layers_list),
            "dim_feedforward": rng.choice(dim_feedforwards),
            "dropout": rng.choice(dropouts),
        }

        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)

    return configs


def run_model(net, name, config, train_loader, val_loader, test_loader, epochs, evaluate_test=False):
    base_name = name.replace("_search", "")
    phase = "final" if evaluate_test else "search"

    run_config = {
        "model_name": base_name,
        "phase": phase,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        **config,
    }

    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=base_name,
        job_type=phase,
        name=f"{name}_{config_to_name(config)}",
        config=run_config,
    ) as run:
        model = Model(net, "adam", "cross_entropy", batch_metrics=["accuracy"])
        model.to(DEVICE)

        param_count = count_parameters(net)

        start_time = time.perf_counter()
        history = model.fit_generator(
            train_loader,
            valid_generator=val_loader,
            epochs=epochs,
            verbose=True,
        )
        train_time = time.perf_counter() - start_time

        acc_key, val_acc_key = get_acc_keys(history)
        best_val_acc = max(h[val_acc_key] for h in history)

        for epoch_idx, h in enumerate(history, start=1):
            run.log(
                {
                    "epoch": epoch_idx,
                    "loss": h["loss"],
                    "val_loss": h["val_loss"],
                    "acc": h[acc_key],
                    "val_acc": h[val_acc_key],
                }
            )

        result = {
            "history": history,
            "best_val_acc": best_val_acc,
            "params": param_count,
            "train_time": train_time,
        }

        run.summary["best_val_acc"] = best_val_acc
        run.summary["params"] = param_count
        run.summary["train_time"] = train_time

        if evaluate_test:
            test_loss, test_acc = model.evaluate_generator(test_loader)
            result["test_loss"] = test_loss
            result["test_acc"] = test_acc

            run.summary["test_loss"] = test_loss
            run.summary["test_acc"] = test_acc

            print(
                f"{name} | best_val_acc={best_val_acc:.4f} | test_acc={test_acc:.4f} "
                f"| params={param_count} | train_time={train_time:.2f}s"
            )
        else:
            print(
                f"{name} | best_val_acc={best_val_acc:.4f} "
                f"| params={param_count} | train_time={train_time:.2f}s"
            )

        return result

def run_best_config():
    # remplie à la main 
    set_seed(42)

    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        wandb.login()

    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    best_config_rnn = {
    }

    best_config_lstm = {
    }

    best_config_transformer = {
    }

    print("final: RNN")
    net = RNNClassifier(num_classes=num_classes, **best_config_rnn)
    run_model(
        net,
        "RNN",
        best_config_rnn,
        train_loader,
        val_loader,
        test_loader,
        FINAL_EPOCHS,
        evaluate_test=True,
    )

    print("final: LSTM")
    net = LSTMClassifier(num_classes=num_classes, **best_config_lstm)
    run_model(
        net,
        "LSTM",
        best_config_lstm,
        train_loader,
        val_loader,
        test_loader,
        FINAL_EPOCHS,
        evaluate_test=True,
    )

    print("final: Transformer")
    net = TransformerClassifier(num_classes=num_classes, **best_config_transformer)
    run_model(
        net,
        "Transformer",
        best_config_transformer,
        train_loader,
        val_loader,
        test_loader,
        FINAL_EPOCHS,
        evaluate_test=True,
    )


evaluate = False
if __name__ == "__main__":
    
    if evaluate:
        run_best_config()
        exit()
        
    set_seed(42)

    wandb_key = os.getenv("WANDB_API_KEY")

    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        wandb.login()

    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    rnn_configs = sample_rnn_lstm_configs(RNN_RANDOM_CONFIGS, seed=42)
    lstm_configs = sample_rnn_lstm_configs(LSTM_RANDOM_CONFIGS, seed=43)
    transformer_configs = sample_transformer_configs(TRANSFORMER_RANDOM_CONFIGS, seed=44)

    print("begin search:")

    for config in rnn_configs:
        print(f"\nTesting RNN config: {config}")
        set_seed(42)
        net = RNNClassifier(num_classes=num_classes, **config)
        run_model(
            net,
            "RNN_search",
            config,
            train_loader,
            val_loader,
            test_loader,
            SEARCH_EPOCHS,
            evaluate_test=False,
        )

    for config in lstm_configs:
        print(f"\nTesting LSTM config: {config}")
        set_seed(42)
        net = LSTMClassifier(num_classes=num_classes, **config)
        run_model(
            net,
            "LSTM_search",
            config,
            train_loader,
            val_loader,
            test_loader,
            SEARCH_EPOCHS,
            evaluate_test=False,
        )

    for config in transformer_configs:
        print(f"\nTesting Transformer config: {config}")
        set_seed(42)
        net = TransformerClassifier(num_classes=num_classes, **config)
        run_model(
            net,
            "Transformer_search",
            config,
            train_loader,
            val_loader,
            test_loader,
            SEARCH_EPOCHS,
            evaluate_test=False,
        )
