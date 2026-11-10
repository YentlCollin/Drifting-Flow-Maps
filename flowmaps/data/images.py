"""Image loading and checkpoint I/O shared by the RGB experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from flowmaps.features.mae import canonical_json_sha256, sha256_file

DATASET_MANIFEST_FORMAT = "driftmodel.celeba64_pixel_dataset_manifest"


DATASET_MANIFEST_VERSION = 1


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return (
            value.detach().cpu().item()
            if value.numel() == 1
            else value.detach().cpu().tolist()
        )
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON provenance cannot contain non-finite floats")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def atomic_json_dump(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def atomic_text_write(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _tree_hash_update(digest: Any, value: Any) -> None:
    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        add(b"tensor")
        add(
            json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        # Reshape scalar optimizer counters before reinterpreting their bytes;
        # ``Tensor.view(dtype)`` rejects zero-dimensional tensors directly.
        add(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        add(b"mapping")
        encoded_keys = []
        for key in value:
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError("tree-hash mapping keys must be JSON scalars")
            encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
            encoded_keys.append((encoded, key))
        for encoded, key in sorted(encoded_keys, key=lambda item: item[0]):
            add(encoded)
            _tree_hash_update(digest, value[key])
        return
    if isinstance(value, tuple):
        add(b"tuple")
        for item in value:
            _tree_hash_update(digest, item)
        return
    if isinstance(value, list):
        add(b"list")
        for item in value:
            _tree_hash_update(digest, item)
        return
    if isinstance(value, torch.device):
        add(b"device")
        add(str(value).encode())
        return
    if isinstance(value, torch.dtype):
        add(b"dtype")
        add(str(value).encode())
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("tree hash rejects non-finite floats")
        add(b"float")
        add(value.hex().encode())
        return
    if isinstance(value, (str, int, bool)) or value is None:
        add(type(value).__name__.encode())
        add(json.dumps(value, separators=(",", ":")).encode())
        return
    raise TypeError(f"unsupported tree-hash value: {type(value).__name__}")


def torch_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256(b"driftmodel-torch-tree-v1\0")
    _tree_hash_update(digest, value)
    return digest.hexdigest()


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        payload = {"timestamp_utc": utc_now(), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()


def _discover_selected_paths(
    data_dir: Path,
    start_index: int,
    end_index: int | None,
) -> list[Path]:
    paths = sorted(
        path for path in data_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no CelebA images in {data_dir}")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("dataset images must be regular files, not symlinks")
    stop = len(paths) if end_index is None else end_index
    if stop > len(paths):
        raise ValueError(
            f"train end index {stop} exceeds the {len(paths)} discovered images"
        )
    selected = paths[start_index:stop]
    if not selected:
        raise ValueError("selected CelebA training split is empty")
    return selected


def build_dataset_manifest(
    data_dir: Path,
    start_index: int,
    end_index: int | None,
) -> dict[str, Any]:
    selected = _discover_selected_paths(data_dir, start_index, end_index)
    records = []
    total_bytes = 0
    for path in selected:
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "relative_path": path.relative_to(data_dir).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    files_sha256 = canonical_json_sha256(records)
    return {
        "format": DATASET_MANIFEST_FORMAT,
        "version": DATASET_MANIFEST_VERSION,
        "root": str(data_dir.resolve()),
        "selection_semantics": "sorted image filenames, Python half-open slice",
        "train_start_index": start_index,
        "train_end_index": start_index + len(selected),
        "image_count": len(selected),
        "total_bytes": total_bytes,
        "image_shape": [3, 64, 64],
        "source_encoding": "RGB uint8 PNG/JPEG decoded by Pillow",
        "files_sha256": files_sha256,
        "files": records,
    }


def _load_rgb_uint8(path: Path) -> Tensor:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        if converted.size != (64, 64):
            raise ValueError(f"CelebA-64 file has size {converted.size}: {path}")
        array = np.array(converted, dtype=np.uint8, copy=True)
    if array.shape != (64, 64, 3):
        raise ValueError(f"unexpected decoded RGB shape {array.shape}: {path}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CelebA64PixelDataset:
    def __init__(
        self,
        data_dir: Path,
        manifest: Mapping[str, Any],
        *,
        preload_uint8: bool,
    ) -> None:
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("dataset manifest lacks files")
        self.paths = [data_dir / str(record["relative_path"]) for record in records]
        self.preloaded: Tensor | None = None
        if preload_uint8:
            self.preloaded = torch.stack([_load_rgb_uint8(path) for path in self.paths])
            if self.preloaded.dtype != torch.uint8:
                raise RuntimeError("preloaded dataset must remain uint8")

    def __len__(self) -> int:
        return len(self.paths)

    def batch(self, indices: Tensor) -> Tensor:
        if indices.device.type != "cpu" or indices.dtype != torch.int64:
            raise ValueError("dataset indices must be CPU int64")
        if self.preloaded is not None:
            return self.preloaded.index_select(0, indices)
        return torch.stack(
            [_load_rgb_uint8(self.paths[index]) for index in indices.tolist()]
        )
