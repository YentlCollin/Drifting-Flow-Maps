import math
from pathlib import Path

import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal


def get_gaussian_mixture_gmm_params(
    dim: int,
    weights,
    mean_offset: float = 2.0,
    std: float = 0.2,
    device=None,
    dtype=torch.float32,
):
    device = torch.device("cpu") if device is None else torch.device(device)
    w = torch.tensor(weights, dtype=dtype, device=device)
    w = w / w.sum()
    means = torch.tensor(
        [[mean_offset] * dim, [-mean_offset / 2.0] * dim], dtype=dtype, device=device
    )
    covs = (std**2.0) * torch.eye(dim, dtype=dtype, device=device).unsqueeze(0).repeat(
        means.shape[0], 1, 1
    )
    return w, means, covs


def get_asymmetric_mixture_gmm_params(
    dim: int,
    weights=None,
    tail_std: float = 0.05,
    warp: bool = True,
    allow_warped_approx: bool = True,
    device=None,
    dtype=torch.float32,
):
    """
    Returns the pre-warp Gaussian-mixture parameters for asymmetric mixture sampling.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    # For warped asymmetric mixtures, return pre-warp GMM parameters as an approximation.
    # This keeps the API usable in interactive/debug flows without throwing.
    _ = (warp, allow_warped_approx)
    device = torch.device("cpu") if device is None else torch.device(device)

    base_means = torch.tensor(
        [
            [2.6, 0.2],
            [-2.1, 1.6],
            [0.0, -3.4],
            [3.7, 3.1],
        ],
        dtype=dtype,
        device=device,
    )
    base_covs = torch.tensor(
        [
            [[0.6, 0.0], [0.0, 0.05]],
            [[0.08, -0.05], [-0.05, 0.4]],
            [[0.2, 0.0], [0.0, 0.2]],
            [[0.3, 0.15], [0.15, 0.3]],
        ],
        dtype=dtype,
        device=device,
    )

    if weights is None:
        w = torch.tensor([0.72, 0.18, 0.08, 0.02], dtype=dtype, device=device)
    else:
        w = torch.tensor(weights, dtype=dtype, device=device)
    w = w / w.sum()

    components = w.numel()
    if components != base_means.shape[0]:
        raise ValueError("weights must match default component count")

    means = torch.zeros(components, dim, dtype=dtype, device=device)
    covs = torch.zeros(components, dim, dim, dtype=dtype, device=device)
    if dim == 1:
        means[:, 0] = base_means[:, 0]
        covs[:, 0, 0] = base_covs[:, 0, 0]
    else:
        means[:, :2] = base_means
        covs[:, :2, :2] = base_covs
        if dim > 2:
            covs[:, 2:, 2:] = (tail_std**2.0) * torch.eye(
                dim - 2, dtype=dtype, device=device
            ).unsqueeze(0)

    return w, means, covs


def sample_gaussian_mixture(
    n: int,
    dim: int,
    weights: list,
    mean_offset: float = 2.0,
    std: float = 0.2,
) -> torch.Tensor:
    weights, means, covs = get_gaussian_mixture_gmm_params(
        dim=dim,
        weights=weights,
        mean_offset=mean_offset,
        std=std,
    )
    mvn = MultivariateNormal(means, covs)
    cat = Categorical(probs=weights)
    return MixtureSameFamily(
        mixture_distribution=cat,
        component_distribution=mvn,
    ).sample((n,))


def sample_asymmetric_mixture(
    n: int,
    dim: int,
    weights=None,
    tail_std: float = 0.05,
    warp: bool = True,
) -> torch.Tensor:
    """Asymmetric mixture with imbalanced weights and anisotropic covariances."""
    weights, means, covs = get_asymmetric_mixture_gmm_params(
        dim=dim,
        weights=weights,
        tail_std=tail_std,
        warp=warp,
        allow_warped_approx=True,
    )

    mvn = MultivariateNormal(means, covs)
    cat = Categorical(probs=weights)
    samples = MixtureSameFamily(
        mixture_distribution=cat,
        component_distribution=mvn,
    ).sample((n,))

    if warp and dim >= 2:
        samples = samples.clone()
        samples[:, 1] = (
            samples[:, 1] + 0.35 * torch.tanh(samples[:, 0]) * samples[:, 0].abs()
        )

    return samples


def sample_asymmetric_mixture_with_params(
    n: int,
    dim: int,
    weights=None,
    tail_std: float = 0.05,
    warp: bool = True,
    return_params: bool = False,
):
    """
    Same sampler as `sample_asymmetric_mixture`, but also returns the pre-warp
    GMM parameters as a dict for easy downstream recovery.
    """
    w, means, covs = get_asymmetric_mixture_gmm_params(
        dim=dim,
        weights=weights,
        tail_std=tail_std,
        warp=warp,
        allow_warped_approx=True,
    )
    mvn = MultivariateNormal(means, covs)
    cat = Categorical(probs=w)
    samples = MixtureSameFamily(
        mixture_distribution=cat,
        component_distribution=mvn,
    ).sample((n,))

    if warp and dim >= 2:
        samples = samples.clone()
        samples[:, 1] = (
            samples[:, 1] + 0.35 * torch.tanh(samples[:, 0]) * samples[:, 0].abs()
        )

    if return_params:
        return samples, {"weights": w, "means": means, "covs": covs}
    return samples


def sample_swiss_roll(n: int, noise: float = 0.03, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    u = torch.rand(n, generator=g)
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    pts = pts / (pts.abs().max() + 1e-8)
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    # Keep an explicit tensor return in case this implementation changes.
    return (
        pts
        if isinstance(pts, torch.Tensor)
        else torch.as_tensor(pts, dtype=torch.float32)
    )


def _next_batch(data_iter, loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)
    return data_iter, batch


def load_mnist_dataset(data_root: Path, train: bool, download: bool):
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        if exc.name != "torchvision":
            raise
        import struct

        import numpy as np

        split = "train" if train else "t10k"
        images_path = Path(data_root) / "MNIST" / "raw" / f"{split}-images-idx3-ubyte"
        labels_path = Path(data_root) / "MNIST" / "raw" / f"{split}-labels-idx1-ubyte"
        if download or not images_path.exists() or not labels_path.exists():
            raise RuntimeError(
                "torchvision is unavailable; download MNIST first or provide its raw IDX files"
            ) from exc
        with images_path.open("rb") as handle:
            magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
            if magic != 2051:
                raise ValueError(
                    f"unexpected image IDX magic number in {images_path}: {magic}"
                )
            images = (
                np.frombuffer(handle.read(), dtype=np.uint8)
                .reshape(count, rows, cols)
                .copy()
            )
        with labels_path.open("rb") as handle:
            magic, count = struct.unpack(">II", handle.read(8))
            if magic != 2049:
                raise ValueError(
                    f"unexpected label IDX magic number in {labels_path}: {magic}"
                )
            labels = np.frombuffer(handle.read(), dtype=np.uint8).reshape(count).copy()
        image_tensor = torch.from_numpy(images).float().unsqueeze(1) / 255.0
        label_tensor = torch.from_numpy(labels).long()
        return torch.utils.data.TensorDataset(image_tensor, label_tensor)

    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(
        root=data_root, train=train, transform=transform, download=download
    )


class ImageFilesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        transform,
        *,
        start_index: int = 0,
        limit: int | None = None,
    ):
        paths = sorted(
            path
            for path in Path(root).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when provided")
        self.paths = paths[start_index : None if limit is None else start_index + limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), 0


class CelebaImageTransform:
    """Deterministic center-crop/resize and tensor conversion for CelebA."""

    def __init__(self, image_size: int = 64):
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.image_size = image_size

    def __call__(self, image):
        import numpy as np
        from PIL import Image

        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )
        array = np.asarray(image, dtype=np.float32)
        if array.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"unexpected CelebA image shape: {array.shape}")
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).div_(255.0)
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("CelebA transform produced invalid pixel values")
        return tensor


def _resolve_celeba_split(data_root: Path, split: str) -> Path:
    data_root = Path(data_root)
    split_aliases = {"train": "training", "val": "validation", "test": "testing"}
    canonical_split = split_aliases.get(split, split)
    candidates = [data_root / canonical_split]
    if canonical_split == "training":
        candidates.append(data_root / "celeba64_train")
    elif canonical_split in {"validation", "testing", "fid_ref"}:
        candidates.append(data_root / "celeba64_fid_ref")
    candidates.append(data_root)
    for candidate in candidates:
        if candidate.is_dir() and any(
            path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for path in candidate.iterdir()
        ):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"no CelebA images found; searched: {searched}")


def load_celeba_dataset(
    data_root: Path,
    split: str = "training",
    *,
    image_size: int = 64,
    start_index: int = 0,
    limit: int | None = None,
):
    """Load common CelebA layouts with preprocessing recorded by the caller."""

    root = _resolve_celeba_split(data_root, split)
    return ImageFilesDataset(
        root,
        CelebaImageTransform(image_size),
        start_index=start_index,
        limit=limit,
    )


@torch.no_grad()
def plot_mnist_grid(
    samples_flat: torch.Tensor, title: str, nrows: int = 8, ncols: int = 8
):
    import matplotlib.pyplot as plt

    count = min(samples_flat.shape[0], nrows * ncols)
    images = samples_flat[:count].view(count, 28, 28).detach().cpu().clamp(0.0, 1.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for idx, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if idx < count:
            ax.imshow(images[idx], cmap="gray", vmin=0.0, vmax=1.0)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def _save_mnist_grid(
    samples_flat: torch.Tensor,
    save_path: Path,
    title: str,
    nrows: int = 8,
    ncols: int = 8,
):
    import matplotlib.pyplot as plt

    count = min(samples_flat.shape[0], nrows * ncols)
    images = samples_flat[:count].view(count, 28, 28).detach().cpu().clamp(0.0, 1.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for idx, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if idx < count:
            ax.imshow(images[idx], cmap="gray", vmin=0.0, vmax=1.0)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def _save_image_grid(
    samples_flat: torch.Tensor,
    save_path: Path,
    title: str,
    *,
    channels: int,
    image_size: int,
    nrows: int = 4,
    ncols: int = 4,
):
    import matplotlib.pyplot as plt

    count = min(samples_flat.shape[0], nrows * ncols)
    images = (
        samples_flat[:count]
        .view(count, channels, image_size, image_size)
        .detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < count:
            ax.imshow(images[idx])
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
