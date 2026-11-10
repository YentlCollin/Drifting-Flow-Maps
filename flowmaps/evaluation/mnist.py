import argparse
import json
import math
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from flowmaps.data.datasets import load_mnist_dataset
from flowmaps.models.toy_mnist import MnistConvGenerator
from flowmaps.paths import REPO_ROOT


class SmallMnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class IdxMnistDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def _read_idx_images(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return torch.from_numpy(data).float().unsqueeze(1) / 255.0


def _read_idx_labels(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n)
    return torch.from_numpy(data).long()


def find_latest_run(run_root: Path) -> Path:
    runs = [p for p in run_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found in {run_root}")
    return sorted(runs, key=lambda p: p.name)[-1]


def parse_config(config_path: Path) -> dict:
    cfg = {}
    if not config_path.exists():
        return cfg
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def get_or_train_classifier(
    data_root: Path,
    device: torch.device,
    download: bool,
    epochs: int,
    batch_size: int,
    num_workers: int,
    force_retrain: bool,
) -> tuple[nn.Module, float]:
    ckpt_path = REPO_ROOT / "runs" / "mnist_eval" / "mnist_classifier.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model = SmallMnistClassifier().to(device)
    train_ds = load_mnist_dataset(data_root=data_root, train=True, download=download)
    test_ds = load_mnist_dataset(data_root=data_root, train=False, download=download)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if ckpt_path.exists() and not force_retrain:
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        acc = evaluate_classifier(model, test_loader, device)
        return model, acc

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    acc = evaluate_classifier(model, test_loader, device)
    torch.save({"model_state_dict": model.state_dict(), "test_acc": acc}, ckpt_path)
    return model, acc


def load_generator_from_run(
    run_dir: Path, device: torch.device
) -> tuple[nn.Module, int, dict]:
    cfg = parse_config(run_dir / "config.txt")
    latent_dim = int(cfg.get("latent_dim", 128))
    base_channels = int(cfg.get("base_channels", 64))

    model = MnistConvGenerator(latent_dim=latent_dim, base_channels=base_channels).to(
        device
    )
    state = torch.load(run_dir / "model_final.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, latent_dim, cfg


def inception_score(
    probabilities: torch.Tensor, splits: int = 10
) -> tuple[float, float]:
    n = probabilities.shape[0]
    splits = max(1, min(splits, n))
    chunk = n // splits
    scores = []
    eps = 1e-8
    for i in range(splits):
        part = (
            probabilities[i * chunk : (i + 1) * chunk]
            if i < splits - 1
            else probabilities[i * chunk :]
        )
        py = part.mean(dim=0, keepdim=True)
        kl = part * (torch.log(part + eps) - torch.log(py + eps))
        scores.append(torch.exp(kl.sum(dim=1).mean()).item())
    if len(scores) == 1:
        return scores[0], 0.0
    return float(torch.tensor(scores).mean().item()), float(
        torch.tensor(scores).std(unbiased=False).item()
    )


def evaluate_generator(
    generator: nn.Module,
    classifier: nn.Module,
    latent_dim: int,
    device: torch.device,
    n_samples: int,
    batch_size: int,
    is_splits: int,
) -> dict:
    probs_all = []
    counts = torch.zeros(10, dtype=torch.long)

    with torch.no_grad():
        n_batches = math.ceil(n_samples / batch_size)
        for i in range(n_batches):
            current_bs = min(batch_size, n_samples - i * batch_size)
            z = torch.randn(current_bs, latent_dim, device=device)
            x = generator(z).view(current_bs, 1, 28, 28)
            logits = classifier(x)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            counts += torch.bincount(preds, minlength=10)
            probs_all.append(probs)

    probs_cat = torch.cat(probs_all, dim=0)
    mode_proportions = (counts.float() / counts.sum().clamp_min(1)).tolist()
    is_mean, is_std = inception_score(probs_cat, splits=is_splits)
    confidence = float(probs_cat.max(dim=1).values.mean().item())

    return {
        "mode_counts": counts.tolist(),
        "mode_proportions": mode_proportions,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "mean_classifier_confidence": confidence,
    }


def real_mnist_proportions(data_root: Path, download: bool) -> list[float]:
    ds = load_mnist_dataset(data_root=data_root, train=False, download=download)
    counts = torch.zeros(10, dtype=torch.long)
    for _, y in ds:
        counts[int(y)] += 1
    return (counts.float() / counts.sum().clamp_min(1)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--classifier-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--force-retrain-classifier", action="store_true")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_root = REPO_ROOT / "data"
    clf, clf_acc = get_or_train_classifier(
        data_root=data_root,
        device=device,
        download=args.download,
        epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_retrain=args.force_retrain_classifier,
    )
    clf.eval()

    latest_drift = find_latest_run(REPO_ROOT / "runs" / "mnist_drift")
    latest_outer_inner = find_latest_run(REPO_ROOT / "runs" / "mnist_drift_outer_inner")

    gen_drift, zdim_drift, _ = load_generator_from_run(latest_drift, device)
    gen_outer, zdim_outer, _ = load_generator_from_run(latest_outer_inner, device)

    result_drift = evaluate_generator(
        generator=gen_drift,
        classifier=clf,
        latent_dim=zdim_drift,
        device=device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        is_splits=args.is_splits,
    )
    result_outer = evaluate_generator(
        generator=gen_outer,
        classifier=clf,
        latent_dim=zdim_outer,
        device=device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        is_splits=args.is_splits,
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "n_samples": args.n_samples,
        "classifier_test_accuracy": clf_acc,
        "real_mnist_proportions_test": real_mnist_proportions(data_root, args.download),
        "runs": {
            "mnist_drift": {
                "latest_run_dir": str(latest_drift),
                **result_drift,
            },
            "mnist_drift_outer_inner": {
                "latest_run_dir": str(latest_outer_inner),
                **result_outer,
            },
        },
    }

    eval_dir = (
        REPO_ROOT / "runs" / "mnist_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Classifier test accuracy: {clf_acc:.4f}")
    print(f"mnist_drift latest: {latest_drift}")
    print(
        f"  Inception score: {result_drift['inception_score_mean']:.4f} +/- {result_drift['inception_score_std']:.4f}"
    )
    print(f"  Mode proportions: {result_drift['mode_proportions']}")
    print(f"mnist_drift_outer_inner latest: {latest_outer_inner}")
    print(
        f"  Inception score: {result_outer['inception_score_mean']:.4f} +/- {result_outer['inception_score_std']:.4f}"
    )
    print(f"  Mode proportions: {result_outer['mode_proportions']}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
