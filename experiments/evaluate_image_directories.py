#!/usr/bin/env python3
"""Compare two image directories with bounded-memory feature metrics.

The evaluator uses the TensorFlow-compatible Inception-v3 implementation from
``pytorch_fid``.  Features are stored as content-addressed ``.npy`` files so a
rerun can memory-map them without loading every activation into RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
CACHE_VERSION = 1
SHA256_CACHE_VERSION = 2
REPORT_VERSION = 1
INCEPTION_FEATURE_DIMENSION = 2048
FEATURE_CACHE_INTEGRITY_MODES = ("metadata", "sha256")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"image directory does not exist: {directory}")
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda path: path.relative_to(directory).as_posix())
    if not files:
        raise ValueError(f"no supported images found under {directory}")
    return files


def select_files(files: Sequence[Path], number: int | None, seed: int) -> list[Path]:
    files = list(files)
    if number is None:
        return files
    if number < 2:
        raise ValueError("num_images must be at least 2")
    if number > len(files):
        raise ValueError(f"requested {number} images, but only {len(files)} are available")
    if number == len(files):
        return files
    indices = sorted(random.Random(seed).sample(range(len(files)), number))
    return [files[index] for index in indices]


def build_file_records(files: Sequence[Path], root: Path) -> list[dict[str, object]]:
    root = root.expanduser().resolve()
    records = []
    for path in files:
        resolved = path.expanduser().resolve()
        records.append(
            {
                "path": resolved.relative_to(root).as_posix(),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return records


def file_manifest_sha256(records: Sequence[dict[str, object]]) -> str:
    return canonical_sha256(list(records))


def exact_overlap_summary(
    real_records: Sequence[dict[str, object]],
    generated_records: Sequence[dict[str, object]],
    *,
    max_examples: int = 20,
) -> dict[str, object]:
    def by_digest(records: Sequence[dict[str, object]]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for record in records:
            result.setdefault(str(record["sha256"]), []).append(str(record["path"]))
        return result

    real_by_hash = by_digest(real_records)
    generated_by_hash = by_digest(generated_records)
    shared = sorted(real_by_hash.keys() & generated_by_hash.keys())
    examples = [
        {
            "sha256": digest,
            "real": real_by_hash[digest],
            "generated": generated_by_hash[digest],
        }
        for digest in shared[:max_examples]
    ]
    return {
        "unique_shared_hashes": len(shared),
        "real_files_in_overlap": sum(len(real_by_hash[digest]) for digest in shared),
        "generated_files_in_overlap": sum(len(generated_by_hash[digest]) for digest in shared),
        "cross_directory_duplicate_pairs": sum(
            len(real_by_hash[digest]) * len(generated_by_hash[digest]) for digest in shared
        ),
        "examples": examples,
    }


class ImagePathDataset(Dataset):
    def __init__(self, files: Sequence[Path]):
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        try:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except Exception as error:
            raise RuntimeError(f"failed to decode image: {path}") from error
        return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)


def inception_image_collate(images: Sequence[torch.Tensor]) -> torch.Tensor:
    shapes = {tuple(image.shape) for image in images}
    if len(shapes) == 1:
        return torch.stack(list(images))
    # InceptionV3 normally performs this resize on the stacked batch.  Resize
    # individually only for mixed-resolution directories, which cannot stack.
    return torch.cat(
        [
            F.interpolate(
                image.unsqueeze(0),
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            )
            for image in images
        ],
        dim=0,
    )


def _load_local_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Inception weights must contain a state dict: {path}")
    return state


def build_inception_extractor(
    device: torch.device,
    *,
    weights_path: Path | None = None,
) -> torch.nn.Module:
    """Build pytorch-fid Inception, optionally forcing an explicit local checkpoint."""
    import pytorch_fid.inception as inception_module

    block = inception_module.InceptionV3.BLOCK_INDEX_BY_DIM[INCEPTION_FEATURE_DIMENSION]
    if weights_path is None:
        model = inception_module.InceptionV3([block])
    else:
        weights_path = weights_path.expanduser().resolve()
        if not weights_path.is_file():
            raise ValueError(f"Inception weights do not exist: {weights_path}")
        state_dict = _load_local_state_dict(weights_path)
        original_loader = inception_module.load_state_dict_from_url
        inception_module.load_state_dict_from_url = lambda *args, **kwargs: state_dict
        try:
            model = inception_module.InceptionV3([block])
        finally:
            inception_module.load_state_dict_from_url = original_loader
    return model.to(device).eval()


@torch.inference_mode()
def iter_feature_batches(
    files: Sequence[Path],
    model: torch.nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Iterator[np.ndarray]:
    loader = DataLoader(
        ImagePathDataset(files),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=inception_image_collate,
    )
    for images in loader:
        output = model(images.to(device, non_blocking=True))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = F.adaptive_avg_pool2d(output, output_size=(1, 1))
        features = output.flatten(start_dim=1)
        yield features.float().cpu().numpy()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_write_feature_batches(
    path: Path,
    batches: Iterable[np.ndarray],
    *,
    expected_count: int,
) -> tuple[int, int]:
    """Write a sequence of feature batches as an atomic, memory-mappable NPY."""
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npy",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    feature_map = None
    offset = 0
    feature_dimension = 0
    try:
        for batch in batches:
            batch = np.asarray(batch, dtype=np.float32)
            if batch.ndim != 2:
                raise ValueError("feature batches must have shape [batch, dimension]")
            if batch.shape[0] == 0:
                continue
            if feature_map is None:
                feature_dimension = int(batch.shape[1])
                feature_map = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.float32,
                    shape=(expected_count, feature_dimension),
                )
            if batch.shape[1] != feature_dimension:
                raise ValueError("feature dimension changed between batches")
            end = offset + batch.shape[0]
            if end > expected_count:
                raise ValueError("feature extractor returned more rows than expected")
            feature_map[offset:end] = batch
            offset = end
        if feature_map is None or offset != expected_count:
            raise ValueError(f"feature extractor returned {offset} rows; expected {expected_count}")
        feature_map.flush()
        feature_map = None
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return expected_count, feature_dimension
    finally:
        if feature_map is not None:
            feature_map = None
        temporary.unlink(missing_ok=True)


def feature_cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    stem = f"inception_{cache_key}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def load_feature_cache(
    cache_dir: Path,
    cache_key: str,
    expected_metadata: dict[str, object],
    *,
    integrity_mode: str = "metadata",
) -> tuple[np.ndarray, dict[str, object]] | None:
    if integrity_mode not in FEATURE_CACHE_INTEGRITY_MODES:
        raise ValueError(f"unsupported feature-cache integrity mode: {integrity_mode}")
    feature_path, metadata_path = feature_cache_paths(cache_dir, cache_key)
    if not feature_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            return None
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if features.ndim != 2 or features.shape[0] != int(expected_metadata["count"]):
            return None
        if metadata.get("feature_dimension") != features.shape[1]:
            return None
        if integrity_mode == "sha256":
            expected_shape = [int(features.shape[0]), int(features.shape[1])]
            if metadata.get("integrity_mode") != "sha256":
                return None
            if metadata.get("dtype") != "float32" or features.dtype != np.dtype(np.float32):
                return None
            if metadata.get("shape") != expected_shape:
                return None
            payload_size = feature_path.stat().st_size
            if metadata.get("payload_size_bytes") != payload_size:
                return None
            payload_sha256 = metadata.get("payload_sha256")
            if (
                not isinstance(payload_sha256, str)
                or len(payload_sha256) != 64
                or sha256_file(feature_path) != payload_sha256
            ):
                return None
            for start in range(0, len(features), 1024):
                if not np.isfinite(np.asarray(features[start : start + 1024])).all():
                    return None
        else:
            payload_sha256 = metadata.get("payload_sha256")
            payload_size = feature_path.stat().st_size
        attestation = {
            "integrity_mode": integrity_mode,
            "cache_key": cache_key,
            "cache_version": metadata.get("cache_version"),
            "payload_path": str(feature_path.resolve()),
            "payload_sha256": payload_sha256,
            "payload_size_bytes": payload_size,
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "dtype": str(features.dtype),
            "shape": [int(features.shape[0]), int(features.shape[1])],
            "sha256_verified": integrity_mode == "sha256",
            "finite_verified": integrity_mode == "sha256",
        }
        return features, attestation
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def load_or_compute_feature_cache(
    cache_dir: Path,
    cache_key: str,
    expected_metadata: dict[str, object],
    batches_factory: Callable[[], Iterable[np.ndarray]],
    *,
    integrity_mode: str = "metadata",
) -> tuple[np.ndarray, bool, dict[str, object]]:
    cached = load_feature_cache(
        cache_dir,
        cache_key,
        expected_metadata,
        integrity_mode=integrity_mode,
    )
    if cached is not None:
        features, attestation = cached
        return features, True, attestation
    feature_path, metadata_path = feature_cache_paths(cache_dir, cache_key)
    count, dimension = atomic_write_feature_batches(
        feature_path,
        batches_factory(),
        expected_count=int(expected_metadata["count"]),
    )
    metadata = {
        **expected_metadata,
        "count": count,
        "feature_dimension": dimension,
        "dtype": "float32",
    }
    if integrity_mode == "sha256":
        metadata.update(
            {
                "integrity_mode": "sha256",
                "shape": [count, dimension],
                "payload_size_bytes": feature_path.stat().st_size,
                "payload_sha256": sha256_file(feature_path),
            }
        )
    atomic_write_json(metadata_path, metadata)
    loaded = load_feature_cache(
        cache_dir,
        cache_key,
        expected_metadata,
        integrity_mode=integrity_mode,
    )
    if loaded is None:
        raise RuntimeError(f"failed to validate newly written feature cache: {feature_path}")
    features, attestation = loaded
    return features, False, attestation


def streaming_mean_covariance(
    features: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("at least two feature vectors are required")
    count = 0
    mean = np.zeros(features.shape[1], dtype=np.float64)
    scatter = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    for start in range(0, features.shape[0], chunk_size):
        batch = np.asarray(features[start : start + chunk_size], dtype=np.float64)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        batch_scatter = centered.T @ centered
        if count == 0:
            mean = batch_mean
            scatter = batch_scatter
            count = batch_count
            continue
        total = count + batch_count
        delta = batch_mean - mean
        scatter += batch_scatter + np.outer(delta, delta) * count * batch_count / total
        mean += delta * batch_count / total
        count = total
    return mean, scatter / (count - 1)


def calculate_fid(
    real_mean: np.ndarray,
    real_covariance: np.ndarray,
    generated_mean: np.ndarray,
    generated_covariance: np.ndarray,
) -> float:
    from pytorch_fid.fid_score import calculate_frechet_distance

    return float(
        calculate_frechet_distance(
            real_mean,
            real_covariance,
            generated_mean,
            generated_covariance,
        )
    )


def polynomial_mmd_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("KID inputs must be two matrices with the same feature dimension")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("unbiased KID requires at least two samples per set")
    scale = float(x.shape[1])
    kernel_xx = (x @ x.T / scale + 1.0) ** 3
    kernel_yy = (y @ y.T / scale + 1.0) ** 3
    kernel_xy = (x @ y.T / scale + 1.0) ** 3
    x_term = (kernel_xx.sum() - np.trace(kernel_xx)) / (x.shape[0] * (x.shape[0] - 1))
    y_term = (kernel_yy.sum() - np.trace(kernel_yy)) / (y.shape[0] * (y.shape[0] - 1))
    return float(x_term + y_term - 2.0 * kernel_xy.mean())


def repeated_kid(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    subset_size: int,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    if repetitions < 2:
        raise ValueError("kid_repetitions must be at least 2 to estimate standard error")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    effective_subset = min(subset_size, len(real_features), len(generated_features))
    if effective_subset < 2:
        raise ValueError("KID subset must contain at least two samples")
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        real_indices = generator.choice(len(real_features), effective_subset, replace=False)
        generated_indices = generator.choice(
            len(generated_features),
            effective_subset,
            replace=False,
        )
        values.append(
            polynomial_mmd_unbiased(
                real_features[real_indices],
                generated_features[generated_indices],
            )
        )
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(repetitions)
    critical_value = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence": confidence,
        "confidence_interval": [
            mean - critical_value * standard_error,
            mean + critical_value * standard_error,
        ],
        "subset_size": effective_subset,
        "repetitions": repetitions,
        "seed": seed,
    }


def sampled_pairwise_distances(
    features: np.ndarray,
    *,
    pair_count: int,
    seed: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("pairwise diversity requires at least two feature vectors")
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    generator = np.random.default_rng(seed)
    first = generator.integers(0, len(features), size=pair_count)
    second = generator.integers(0, len(features) - 1, size=pair_count)
    second += second >= first
    distances = np.empty(pair_count, dtype=np.float64)
    for start in range(0, pair_count, chunk_size):
        end = min(start + chunk_size, pair_count)
        delta = (
            np.asarray(features[first[start:end]], dtype=np.float64)
            - np.asarray(features[second[start:end]], dtype=np.float64)
        )
        distances[start:end] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return distances


def diversity_summary(
    features: np.ndarray,
    covariance: np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> dict[str, float | int]:
    distances = sampled_pairwise_distances(
        features,
        pair_count=pair_count,
        seed=seed,
    )
    covariance_trace = float(np.trace(covariance))
    covariance_frobenius_squared = float(np.square(covariance).sum())
    participation_ratio = (
        covariance_trace**2 / covariance_frobenius_squared
        if covariance_frobenius_squared > 0
        else 0.0
    )
    return {
        "covariance_trace": covariance_trace,
        "covariance_frobenius_norm": math.sqrt(covariance_frobenius_squared),
        "covariance_participation_ratio": participation_ratio,
        "sampled_pair_distance_mean": float(distances.mean()),
        "sampled_pair_distance_std": float(distances.std(ddof=1)) if pair_count > 1 else 0.0,
        "sampled_pair_distance_median": float(np.median(distances)),
        "sampled_pair_count": pair_count,
    }


def nearest_neighbor_distances(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return exact cross-set nearest distances using O(chunk_size^2) memory."""
    if query_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("nearest-neighbor inputs must be matrices")
    if query_features.shape[1] != reference_features.shape[1]:
        raise ValueError("nearest-neighbor feature dimensions must match")
    if len(query_features) == 0 or len(reference_features) == 0:
        raise ValueError("nearest-neighbor sets must be non-empty")
    if chunk_size < 1:
        raise ValueError("nearest-neighbor chunk_size must be positive")

    output = np.empty(len(query_features), dtype=np.float32)
    for query_start in range(0, len(query_features), chunk_size):
        query_end = min(query_start + chunk_size, len(query_features))
        query = torch.as_tensor(
            np.array(query_features[query_start:query_end], copy=True),
            dtype=torch.float32,
            device=device,
        )
        query_norm = query.square().sum(dim=1, keepdim=True)
        minimum = torch.full((len(query),), torch.inf, device=device)
        for reference_start in range(0, len(reference_features), chunk_size):
            reference_end = min(reference_start + chunk_size, len(reference_features))
            reference = torch.as_tensor(
                np.array(reference_features[reference_start:reference_end], copy=True),
                dtype=torch.float32,
                device=device,
            )
            squared = (
                query_norm
                + reference.square().sum(dim=1).unsqueeze(0)
                - 2.0 * query @ reference.T
            ).clamp_min_(0.0)
            minimum = torch.minimum(minimum, squared.min(dim=1).values)
        output[query_start:query_end] = minimum.sqrt().cpu().numpy()
    return output


def distance_summary(distances: np.ndarray) -> dict[str, float | int]:
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "count": len(distances),
        "mean": float(distances.mean()),
        "standard_deviation": float(distances.std(ddof=1)) if len(distances) > 1 else 0.0,
        "minimum": float(distances.min()),
        "p05": float(np.quantile(distances, 0.05)),
        "median": float(np.median(distances)),
        "p95": float(np.quantile(distances, 0.95)),
        "maximum": float(distances.max()),
    }


def _format_number(value: object) -> str:
    return f"{float(value):.6g}"


def markdown_report(payload: dict[str, object]) -> str:
    metrics = payload["metrics"]
    provenance = payload["provenance"]
    kid = metrics["kid"]
    interval = kid["confidence_interval"]
    diversity = metrics["diversity"]
    nearest = metrics["nearest_neighbor"]
    overlap = metrics["exact_overlap"]
    lines = [
        "# Image-directory evaluation",
        "",
        "## Distribution metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| FID | {_format_number(metrics['fid'])} |",
        f"| KID | {_format_number(kid['mean'])} ± {_format_number(kid['standard_error'])} SE |",
        (
            f"| KID {100 * float(kid['confidence']):.1f}% CI | "
            f"[{_format_number(interval[0])}, {_format_number(interval[1])}] |"
        ),
        "",
        "## Diversity",
        "",
        "| set | covariance trace | covariance participation ratio | sampled pair distance |",
        "|---|---:|---:|---:|",
    ]
    for name in ("real", "generated"):
        row = diversity[name]
        lines.append(
            f"| {name} | {_format_number(row['covariance_trace'])} | "
            f"{_format_number(row['covariance_participation_ratio'])} | "
            f"{_format_number(row['sampled_pair_distance_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Cross-set nearest neighbors",
            "",
            "| direction | mean | median | p95 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("real_to_generated", "generated_to_real"):
        row = nearest[name]
        lines.append(
            f"| {name.replace('_', ' ')} | {_format_number(row['mean'])} | "
            f"{_format_number(row['median'])} | {_format_number(row['p95'])} |"
        )
    lines.extend(
        [
            "",
            "## Exact byte overlap",
            "",
            f"- Shared SHA256 hashes: {overlap['unique_shared_hashes']}",
            f"- Real files involved: {overlap['real_files_in_overlap']}",
            f"- Generated files involved: {overlap['generated_files_in_overlap']}",
            f"- Cross-directory duplicate pairs: {overlap['cross_directory_duplicate_pairs']}",
            "",
            "## Provenance",
            "",
            f"- Evaluation SHA256: `{provenance['evaluation_sha256']}`",
            f"- Options SHA256: `{provenance['options_sha256']}`",
            f"- Real file-list SHA256: `{provenance['real']['manifest_sha256']}`",
            f"- Generated file-list SHA256: `{provenance['generated']['manifest_sha256']}`",
            f"- Real directory: `{provenance['real']['directory']}`",
            f"- Generated directory: `{provenance['generated']['directory']}`",
            f"- Selected counts: {provenance['real']['count']} real, "
            f"{provenance['generated']['count']} generated",
            f"- Inception weights: `{provenance['feature_extractor']['weights']}`",
        ]
    )
    for name in ("real", "generated"):
        cache = provenance[name]["feature_cache_attestation"]
        lines.extend(
            [
                f"- {name.capitalize()} cache integrity: `{cache['integrity_mode']}`",
                f"- {name.capitalize()} cache payload SHA256: `{cache['payload_sha256']}`",
                f"- {name.capitalize()} cache metadata SHA256: `{cache['metadata_sha256']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def extractor_provenance(weights_path: Path | None) -> dict[str, object]:
    import pytorch_fid
    from pytorch_fid.inception import FID_WEIGHTS_URL

    if weights_path is None:
        weights = FID_WEIGHTS_URL
        weights_sha256 = None
    else:
        weights_path = weights_path.expanduser().resolve()
        if not weights_path.is_file():
            raise ValueError(f"Inception weights do not exist: {weights_path}")
        weights = str(weights_path)
        weights_sha256 = sha256_file(weights_path)
    return {
        "name": "pytorch_fid.InceptionV3",
        "pytorch_fid_version": getattr(pytorch_fid, "__version__", "unknown"),
        "feature_dimension": INCEPTION_FEATURE_DIMENSION,
        "weights": weights,
        "weights_sha256": weights_sha256,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    real_dir = args.real_dir.expanduser().resolve()
    generated_dir = args.generated_dir.expanduser().resolve()
    real_files = select_files(discover_images(real_dir), args.num_images, args.seed)
    generated_files = select_files(discover_images(generated_dir), args.num_images, args.seed)
    real_records = build_file_records(real_files, real_dir)
    generated_records = build_file_records(generated_files, generated_dir)
    real_manifest = file_manifest_sha256(real_records)
    generated_manifest = file_manifest_sha256(generated_records)

    device = resolve_device(args.device)
    if args.torch_cache_dir is not None:
        torch_cache_dir = args.torch_cache_dir.expanduser().resolve()
        torch_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(torch_cache_dir))
    extractor = extractor_provenance(args.inception_weights)
    options = {
        "num_images": args.num_images,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": str(device),
        "num_workers": args.num_workers,
        "kid_subset_size": args.kid_subset_size,
        "kid_repetitions": args.kid_repetitions,
        "confidence": args.confidence,
        "diversity_pairs": args.diversity_pairs,
        "statistics_chunk_size": args.statistics_chunk_size,
        "nearest_neighbor_chunk_size": args.nearest_neighbor_chunk_size,
        "feature_cache_integrity": args.feature_cache_integrity,
        "feature_extractor": extractor,
    }
    options_sha256 = canonical_sha256(options)
    evaluation_sha256 = canonical_sha256(
        {
            "real_manifest_sha256": real_manifest,
            "generated_manifest_sha256": generated_manifest,
            "options_sha256": options_sha256,
        }
    )

    cache_dir = args.feature_cache_dir.expanduser().resolve()
    model: torch.nn.Module | None = None

    def batches_for(files: Sequence[Path]) -> Iterable[np.ndarray]:
        nonlocal model
        if model is None:
            model = build_inception_extractor(device, weights_path=args.inception_weights)
        return iter_feature_batches(
            files,
            model,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )

    def cached_features(
        files: Sequence[Path],
        manifest_sha256: str,
    ) -> tuple[np.ndarray, bool, str, dict[str, object]]:
        metadata = {
            "cache_version": (
                SHA256_CACHE_VERSION
                if args.feature_cache_integrity == "sha256"
                else CACHE_VERSION
            ),
            "count": len(files),
            "manifest_sha256": manifest_sha256,
            "extractor": extractor,
        }
        if args.feature_cache_integrity == "sha256":
            metadata["integrity_mode"] = "sha256"
        key = canonical_sha256(metadata)
        features, hit, attestation = load_or_compute_feature_cache(
            cache_dir,
            key,
            metadata,
            lambda: batches_for(files),
            integrity_mode=args.feature_cache_integrity,
        )
        return features, hit, key, attestation

    real_features, real_cache_hit, real_cache_key, real_cache_attestation = cached_features(
        real_files,
        real_manifest,
    )
    (
        generated_features,
        generated_cache_hit,
        generated_cache_key,
        generated_cache_attestation,
    ) = cached_features(generated_files, generated_manifest)
    real_mean, real_covariance = streaming_mean_covariance(
        real_features,
        chunk_size=args.statistics_chunk_size,
    )
    generated_mean, generated_covariance = streaming_mean_covariance(
        generated_features,
        chunk_size=args.statistics_chunk_size,
    )
    fid = calculate_fid(real_mean, real_covariance, generated_mean, generated_covariance)
    kid = repeated_kid(
        real_features,
        generated_features,
        subset_size=args.kid_subset_size,
        repetitions=args.kid_repetitions,
        seed=args.seed,
        confidence=args.confidence,
    )
    kid["kernel"] = {
        "name": "polynomial",
        "degree": 3,
        "gamma": "1 / feature_dimension",
        "coef0": 1.0,
        "estimator": "unbiased_off_diagonal",
    }
    diversity = {
        "real": diversity_summary(
            real_features,
            real_covariance,
            pair_count=args.diversity_pairs,
            seed=args.seed + 1,
        ),
        "generated": diversity_summary(
            generated_features,
            generated_covariance,
            pair_count=args.diversity_pairs,
            seed=args.seed + 2,
        ),
    }
    real_to_generated = nearest_neighbor_distances(
        real_features,
        generated_features,
        chunk_size=args.nearest_neighbor_chunk_size,
        device=device,
    )
    generated_to_real = nearest_neighbor_distances(
        generated_features,
        real_features,
        chunk_size=args.nearest_neighbor_chunk_size,
        device=device,
    )
    payload: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "fid": fid,
            "kid": kid,
            "diversity": diversity,
            "nearest_neighbor": {
                "feature_space": "pytorch_fid_inception_pool3_euclidean",
                "real_to_generated": distance_summary(real_to_generated),
                "generated_to_real": distance_summary(generated_to_real),
            },
            "exact_overlap": exact_overlap_summary(real_records, generated_records),
        },
        "provenance": {
            "evaluation_sha256": evaluation_sha256,
            "options_sha256": options_sha256,
            "options": options,
            "real": {
                "directory": str(real_dir),
                "count": len(real_files),
                "manifest_sha256": real_manifest,
                "feature_cache_key": real_cache_key,
                "feature_cache_hit": real_cache_hit,
                "feature_cache_attestation": real_cache_attestation,
            },
            "generated": {
                "directory": str(generated_dir),
                "count": len(generated_files),
                "manifest_sha256": generated_manifest,
                "feature_cache_key": generated_cache_key,
                "feature_cache_hit": generated_cache_hit,
                "feature_cache_attestation": generated_cache_attestation,
            },
            "feature_extractor": extractor,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    }
    atomic_write_json(args.output_json.expanduser().resolve(), payload)
    atomic_write_text(
        args.output_markdown.expanduser().resolve(),
        markdown_report(payload),
    )
    return payload


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_dir", type=Path)
    parser.add_argument("generated_dir", type=Path)
    parser.add_argument("--num-images", "--number", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch", type=positive_int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", "--workers", type=int, default=4)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "driftmodel" / "inception_features",
    )
    parser.add_argument(
        "--feature-cache-integrity",
        choices=FEATURE_CACHE_INTEGRITY_MODES,
        default="metadata",
        help=(
            "Cache validation mode. 'sha256' records and verifies the complete NPY payload, "
            "its size, dtype, shape, and finiteness; 'metadata' preserves legacy behavior."
        ),
    )
    parser.add_argument(
        "--inception-weights",
        type=Path,
        help="Local pytorch-fid Inception checkpoint; otherwise the torch hub cache is used.",
    )
    parser.add_argument(
        "--torch-cache-dir",
        type=Path,
        help="Optional torch hub cache directory for downloaded Inception weights.",
    )
    parser.add_argument("--kid-subset-size", type=positive_int, default=1000)
    parser.add_argument("--kid-repetitions", type=positive_int, default=100)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--diversity-pairs", type=positive_int, default=10_000)
    parser.add_argument("--statistics-chunk-size", type=positive_int, default=1024)
    parser.add_argument("--nearest-neighbor-chunk-size", type=positive_int, default=512)
    parser.add_argument("--output-json", type=Path, default=Path("image_directory_metrics.json"))
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("image_directory_metrics.md"),
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.num_images is not None and args.num_images < 2:
        parser.error("--num-images must be at least 2")
    if args.kid_repetitions < 2:
        parser.error("--kid-repetitions must be at least 2")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie strictly between 0 and 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = evaluate(args)
    print(f"FID: {payload['metrics']['fid']:.6g}")
    print(f"JSON: {args.output_json.expanduser().resolve()}")
    print(f"Markdown: {args.output_markdown.expanduser().resolve()}")


if __name__ == "__main__":
    main()
