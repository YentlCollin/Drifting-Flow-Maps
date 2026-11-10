"""Latent-space building blocks for the CelebA WGF flow-map experiments.

The image autoencoder is deliberately kept outside the WGF objective.  CelebA
images in ``[0, 1]`` are encoded once into scaled Stable-Diffusion latents and
the existing losses in :mod:`flowmaps.objectives.wgf` then operate on flattened Euclidean
vectors of size 256.  In symbols, nothing in the theory changes: ``x``, the
reference particles and the data particles simply live in the VAE latent
space instead of the 12,288-dimensional pixel space.

``diffusers`` is imported lazily, so importing this module and training the
flow model do not require it once a latent cache has been built.  Tests and
offline users may inject any VAE exposing the small ``AutoencoderKL`` API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import flowmaps.data.cache_io as official_runtime

DEFAULT_VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
DEFAULT_VAE_SCALING_FACTOR = 0.18215

CELEBA_IMAGE_SHAPE = (3, 64, 64)
CELEBA_LATENT_SHAPE = (4, 8, 8)
CELEBA_LATENT_DIM = math.prod(CELEBA_LATENT_SHAPE)

LATENT_CACHE_FORMAT = "driftmodel.celeba_latents"
LATENT_CACHE_VERSION = 1
LATENT_CACHE_ENCODINGS = ("posterior_mean", "posterior_sample")
DEFAULT_POSTERIOR_SAMPLE_SEED = 42
POSTERIOR_SAMPLE_RNG_SCHEME = "sha256_fold_in_index_torch_cpu_v1"
POSTERIOR_SAMPLE_INDEXING = "contiguous_absolute_dataset_index"
LATENT_PRIOR_FORMAT = "driftmodel.celeba_latent_gaussian_prior"
LATENT_PRIOR_VERSION = 1
LATENT_PRIOR_FILENAME = "latent_prior.pt"


def _validate_posterior_sample_coordinates(seed: int, index_start: int) -> None:
    for name, value in (
        ("posterior_sample_seed", seed),
        ("posterior_sample_index_start", index_start),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _posterior_sample_image_seed(seed: int, absolute_index: int) -> int:
    """Fold one absolute image index into a stable CPU-generator seed."""

    payload = f"{POSTERIOR_SAMPLE_RNG_SCHEME}:{seed}:{absolute_index}".encode("ascii")
    # torch.Generator.manual_seed accepts signed 64-bit positive seeds.  Hashing
    # provides independent per-image streams without depending on loader batches.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash the bytes of one regular file (following a file symlink)."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_file_hashes(directory: str | Path) -> dict[str, str]:
    """Hash every file below a local model directory, including symlink targets."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash a tensor's canonical CPU bytes together with its dtype and shape."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _valid_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class LatentGaussianSampler:
    """Device-local sampler prepared from a :class:`LatentGaussianPrior`."""

    mode: str
    mean: torch.Tensor
    cholesky: torch.Tensor

    def transform_standard(
        self, samples: torch.Tensor, *, rowwise: bool = False
    ) -> torch.Tensor:
        """Map standard-normal rows to the configured latent prior.

        ``rowwise=True`` deliberately uses one matrix-vector product per
        sample. Standalone image generation uses that path so changing the
        outer generation batch partition cannot change floating-point
        reduction choices.
        """

        if samples.ndim != 2 or samples.shape[1] != CELEBA_LATENT_DIM:
            raise ValueError(
                f"expected standard samples shaped [B, {CELEBA_LATENT_DIM}], "
                f"got {tuple(samples.shape)}"
            )
        if self.mode == "standard":
            return samples
        original_device, original_dtype = samples.device, samples.dtype
        values = samples.to(device=self.mean.device, dtype=self.mean.dtype)
        if rowwise:
            transformed = torch.stack(
                [torch.mv(self.cholesky, row) + self.mean for row in values],
                dim=0,
            )
        else:
            transformed = values.matmul(self.cholesky.T) + self.mean
        return transformed.to(device=original_device, dtype=original_dtype)


@dataclass(frozen=True)
class LatentGaussianPrior:
    """A reproducible standard or train-only full-covariance latent prior.

    Fitted tensors are retained in CPU float64, matching the Gaussian control
    experiment's population-moment estimator. Actual training, diagnostics,
    and standalone generation all use the same explicit float32 sampling
    tensors; the float64 state remains the auditable fit definition.
    """

    mode: str
    mean: torch.Tensor
    covariance: torch.Tensor
    cholesky: torch.Tensor
    sample_count: int | None = None
    train_start_index: int | None = None
    train_end_index: int | None = None
    source_values_sha256: str | None = None
    source_cache_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"standard", "train_full_gaussian"}:
            raise ValueError(
                "latent prior mode must be 'standard' or 'train_full_gaussian'"
            )
        if self.mode == "standard":
            if any(
                tensor.numel() for tensor in (self.mean, self.covariance, self.cholesky)
            ):
                raise ValueError("standard latent prior must not store dense tensors")
            return
        if self.mean.shape != (CELEBA_LATENT_DIM,):
            raise ValueError(
                f"latent prior mean must have shape ({CELEBA_LATENT_DIM},)"
            )
        expected_matrix_shape = (CELEBA_LATENT_DIM, CELEBA_LATENT_DIM)
        if (
            self.covariance.shape != expected_matrix_shape
            or self.cholesky.shape != expected_matrix_shape
        ):
            raise ValueError(
                f"latent prior matrices must have shape {expected_matrix_shape}"
            )
        if any(
            tensor.device.type != "cpu" or tensor.dtype != torch.float64
            for tensor in (self.mean, self.covariance, self.cholesky)
        ):
            raise ValueError("fitted latent prior tensors must be CPU float64")
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (self.mean, self.covariance, self.cholesky)
        ):
            raise ValueError("latent prior tensors must be finite")
        if not torch.equal(self.cholesky, torch.tril(self.cholesky)):
            raise ValueError("latent prior Cholesky factor must be lower triangular")
        if not bool((self.cholesky.diagonal() > 0).all()):
            raise ValueError("latent prior Cholesky factor must have positive diagonal")
        if not torch.allclose(self.covariance, self.covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("latent prior covariance must be symmetric")
        reconstructed = self.cholesky.matmul(self.cholesky.T)
        if not torch.allclose(self.covariance, reconstructed, rtol=1e-10, atol=1e-12):
            raise ValueError("latent prior covariance and Cholesky factor disagree")
        if self.sample_count is None or self.sample_count < 2:
            raise ValueError("fitted latent prior sample_count must be at least two")
        if self.train_start_index is None or self.train_end_index is None:
            raise ValueError("fitted latent prior requires explicit training indices")
        if (
            self.train_start_index < 0
            or self.train_end_index <= self.train_start_index
            or self.train_end_index - self.train_start_index != self.sample_count
        ):
            raise ValueError("latent prior training indices disagree with sample_count")
        if not _valid_sha256(self.source_values_sha256):
            raise ValueError(
                "fitted latent prior requires a valid source-values SHA-256"
            )
        if not _valid_sha256(self.source_cache_sha256):
            raise ValueError(
                "fitted latent prior requires a valid source-cache SHA-256"
            )

    @classmethod
    def standard(cls) -> "LatentGaussianPrior":
        empty = torch.empty(0, dtype=torch.float64)
        return cls("standard", empty, empty, empty)

    @classmethod
    def fit_full(
        cls,
        flat_latents: torch.Tensor,
        *,
        train_start_index: int,
        train_end_index: int,
        source_cache_sha256: str,
    ) -> "LatentGaussianPrior":
        unflatten_latents(flat_latents)  # validate the fixed CelebA coordinate system
        if train_end_index - train_start_index != flat_latents.shape[0]:
            raise ValueError(
                "latent prior fit indices disagree with the supplied training tensor"
            )
        values = (
            flat_latents.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        mean = values.mean(dim=0)
        centered = values - mean
        covariance = centered.T.matmul(centered).div_(values.shape[0])
        try:
            cholesky = torch.linalg.cholesky(covariance)
        except RuntimeError as error:
            raise ValueError(
                "train-only latent covariance is not positive definite; "
                "the canonical full-Gaussian prior does not add jitter or shrinkage"
            ) from error
        return cls(
            "train_full_gaussian",
            mean.contiguous(),
            covariance.contiguous(),
            cholesky.contiguous(),
            int(values.shape[0]),
            int(train_start_index),
            int(train_end_index),
            sha256_tensor(unflatten_latents(values)),
            source_cache_sha256,
        )

    def sampler(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> LatentGaussianSampler:
        if not dtype.is_floating_point:
            raise ValueError("latent prior sampler dtype must be floating point")
        if self.mode == "standard":
            empty = torch.empty(0, device=device, dtype=dtype)
            return LatentGaussianSampler("standard", empty, empty)
        return LatentGaussianSampler(
            self.mode,
            self.mean.to(device=device, dtype=dtype),
            self.cholesky.to(device=device, dtype=dtype),
        )

    def metadata(self) -> dict[str, Any]:
        if self.mode == "standard":
            return {"mode": "standard", "distribution": "N(0,I)"}
        eigenvalues = torch.linalg.eigvalsh(self.covariance)
        sampling_mean = self.mean.to(torch.float32)
        sampling_cholesky = self.cholesky.to(torch.float32)
        return {
            "mode": "train_full_gaussian",
            "kind": "full_moment_matched_gaussian",
            "distribution": "N(train_mean,train_covariance)",
            "formula": "x = mean + chol(covariance) z, z ~ N(0,I_256)",
            "fit_start_inclusive": self.train_start_index,
            "fit_stop_exclusive": self.train_end_index,
            "fit_count": self.sample_count,
            "estimator": "population_moments_correction_0",
            "fit_tensor_dtype": "torch.float64",
            "sampling_tensor_dtype": "torch.float32",
            "source_values_sha256": self.source_values_sha256,
            "source_cache_sha256": self.source_cache_sha256,
            "mean_sha256": sha256_tensor(self.mean),
            "covariance_sha256": sha256_tensor(self.covariance),
            "cholesky_sha256": sha256_tensor(self.cholesky),
            "sampling_mean_sha256": sha256_tensor(sampling_mean),
            "sampling_cholesky_sha256": sha256_tensor(sampling_cholesky),
            "mean_norm": float(self.mean.norm()),
            "covariance_trace": float(eigenvalues.sum()),
            "eigenvalue_min": float(eigenvalues.min()),
            "eigenvalue_max": float(eigenvalues.max()),
            "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LATENT_PRIOR_FORMAT,
            "version": LATENT_PRIOR_VERSION,
            "metadata": self.metadata(),
            "mean": self.mean.detach().cpu().contiguous(),
            "covariance": self.covariance.detach().cpu().contiguous(),
            "cholesky": self.cholesky.detach().cpu().contiguous(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LatentGaussianPrior":
        if (
            not isinstance(payload, Mapping)
            or payload.get("format") != LATENT_PRIOR_FORMAT
        ):
            raise ValueError("not a driftmodel CelebA latent-prior sidecar")
        if payload.get("version") != LATENT_PRIOR_VERSION:
            raise ValueError(
                f"unsupported latent-prior version: {payload.get('version')}"
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("latent-prior sidecar lacks metadata")
        mode = str(metadata.get("mode", ""))
        tensors = [payload.get(key) for key in ("mean", "covariance", "cholesky")]
        if not all(torch.is_tensor(tensor) for tensor in tensors):
            raise ValueError("latent-prior sidecar lacks tensor state")
        if mode == "standard":
            prior = cls.standard()
        elif mode == "train_full_gaussian":
            prior = cls(
                mode,
                tensors[0].detach().cpu().contiguous(),
                tensors[1].detach().cpu().contiguous(),
                tensors[2].detach().cpu().contiguous(),
                int(metadata["fit_count"]),
                int(metadata["fit_start_inclusive"]),
                int(metadata["fit_stop_exclusive"]),
                str(metadata["source_values_sha256"]),
                str(metadata["source_cache_sha256"]),
            )
        else:
            raise ValueError(f"unsupported latent-prior mode in sidecar: {mode!r}")
        if not prior.matches_metadata(metadata):
            raise ValueError("latent-prior metadata or tensor hashes are inconsistent")
        return prior

    def matches_metadata(self, candidate: Mapping[str, Any]) -> bool:
        """Validate stored metadata without bitwise-comparing derived spectra.

        Tensor hashes, fit provenance, and all categorical fields remain exact.
        Spectral summaries are diagnostics derived by threaded linear algebra;
        their last bits may legitimately differ across BLAS thread counts.
        """

        if not isinstance(candidate, Mapping):
            return False
        expected = self.metadata()
        candidate = dict(candidate)
        if candidate.keys() != expected.keys():
            return False
        derived_float_keys = {
            "mean_norm",
            "covariance_trace",
            "eigenvalue_min",
            "eigenvalue_max",
            "condition_number",
        }
        for key, expected_value in expected.items():
            candidate_value = candidate[key]
            if key in derived_float_keys:
                if (
                    isinstance(candidate_value, bool)
                    or not isinstance(candidate_value, (int, float))
                    or not math.isclose(
                        float(candidate_value),
                        float(expected_value),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    return False
            elif candidate_value != expected_value:
                return False
        return True

    def same_definition(self, other: "LatentGaussianPrior") -> bool:
        return (
            self.mode == other.mode
            and self.sample_count == other.sample_count
            and self.train_start_index == other.train_start_index
            and self.train_end_index == other.train_end_index
            and self.source_values_sha256 == other.source_values_sha256
            and self.source_cache_sha256 == other.source_cache_sha256
            and all(
                torch.equal(left, right)
                for left, right in zip(
                    (self.mean, self.covariance, self.cholesky),
                    (other.mean, other.covariance, other.cholesky),
                    strict=True,
                )
            )
        )


@dataclass(frozen=True)
class LatentAffineNormalization:
    """Affine coordinate change for flattened CelebA VAE latents.

    Channel statistics are shared over samples and the 8x8 spatial grid.  The
    class deliberately operates on flattened particles so the WGF code stays
    unaware of the VAE layout.
    """

    mode: str
    mean: torch.Tensor
    std: torch.Tensor
    sample_count: int | None = None
    train_start_index: int | None = None
    train_end_index: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "channel"}:
            raise ValueError("latent normalization mode must be 'none' or 'channel'")
        if self.mean.shape != (CELEBA_LATENT_SHAPE[0],) or self.std.shape != (
            CELEBA_LATENT_SHAPE[0],
        ):
            raise ValueError(
                "latent normalization mean/std must contain one value per channel"
            )
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.std).all()):
            raise ValueError("latent normalization statistics must be finite")
        if not bool((self.std > 0).all()):
            raise ValueError(
                "latent normalization standard deviations must be positive"
            )
        if self.sample_count is not None and self.sample_count < 1:
            raise ValueError("latent normalization sample_count must be positive")
        if (self.train_start_index is None) != (self.train_end_index is None):
            raise ValueError(
                "latent normalization training indices must be both present or both absent"
            )
        if self.train_start_index is not None:
            if (
                self.train_start_index < 0
                or self.train_end_index <= self.train_start_index
            ):
                raise ValueError("invalid latent normalization training indices")
            if (
                self.sample_count is not None
                and self.train_end_index - self.train_start_index != self.sample_count
            ):
                raise ValueError(
                    "latent normalization indices disagree with sample_count"
                )

    @classmethod
    def identity(cls) -> "LatentAffineNormalization":
        return cls("none", torch.zeros(4), torch.ones(4), None, None, None)

    @classmethod
    def fit_channel(
        cls,
        flat_latents: torch.Tensor,
        *,
        train_start_index: int | None = None,
        train_end_index: int | None = None,
    ) -> "LatentAffineNormalization":
        spatial = unflatten_latents(flat_latents)
        # Accumulate in float64 so the saved constants are independent of the
        # training/device dtype and stable across later resume invocations.
        values = spatial.detach().to(device="cpu", dtype=torch.float64)
        mean = values.mean(dim=(0, 2, 3))
        std = values.var(dim=(0, 2, 3), correction=0).sqrt()
        if not bool((std > torch.finfo(std.dtype).eps).all()):
            raise ValueError("cannot standardize a latent channel with zero variance")
        return cls(
            "channel",
            mean,
            std,
            int(flat_latents.shape[0]),
            train_start_index,
            train_end_index,
        )

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, Any] | None
    ) -> "LatentAffineNormalization":
        if metadata is None:
            return cls.identity()
        mode = str(metadata.get("mode", "none"))
        if mode == "none":
            return cls.identity()
        if mode != "channel":
            raise ValueError(f"unsupported latent normalization mode: {mode!r}")
        if "mean" not in metadata or "std" not in metadata:
            raise ValueError("channel latent normalization requires saved mean and std")
        return cls(
            mode,
            torch.as_tensor(metadata["mean"], dtype=torch.float64),
            torch.as_tensor(metadata["std"], dtype=torch.float64),
            int(metadata["sample_count"])
            if metadata.get("sample_count") is not None
            else None,
            (
                int(metadata["train_start_index"])
                if metadata.get("train_start_index") is not None
                else None
            ),
            (
                int(metadata["train_end_index"])
                if metadata.get("train_end_index") is not None
                else None
            ),
        )

    def normalize(self, flat_latents: torch.Tensor) -> torch.Tensor:
        unflatten_latents(flat_latents)  # shape validation
        if self.mode == "none":
            return flat_latents
        mean = self.mean.to(device=flat_latents.device, dtype=flat_latents.dtype).view(
            1, 4, 1, 1
        )
        std = self.std.to(device=flat_latents.device, dtype=flat_latents.dtype).view(
            1, 4, 1, 1
        )
        spatial = flat_latents.reshape(flat_latents.shape[0], *CELEBA_LATENT_SHAPE)
        return ((spatial - mean) / std).reshape_as(flat_latents)

    def denormalize(self, flat_latents: torch.Tensor) -> torch.Tensor:
        unflatten_latents(flat_latents)  # shape validation
        if self.mode == "none":
            return flat_latents
        mean = self.mean.to(device=flat_latents.device, dtype=flat_latents.dtype).view(
            1, 4, 1, 1
        )
        std = self.std.to(device=flat_latents.device, dtype=flat_latents.dtype).view(
            1, 4, 1, 1
        )
        spatial = flat_latents.reshape(flat_latents.shape[0], *CELEBA_LATENT_SHAPE)
        return (spatial * std + mean).reshape_as(flat_latents)

    def metadata(
        self,
        *,
        train_start_index: int | None = None,
        train_end_index: int | None = None,
    ) -> dict[str, Any]:
        if self.mode == "none":
            return {"mode": "none"}
        return {
            "mode": "channel",
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "sample_count": self.sample_count,
            "elements_per_channel": int(self.sample_count or 0) * 8 * 8,
            "variance_correction": 0,
            "train_start_index": (
                self.train_start_index
                if train_start_index is None
                else train_start_index
            ),
            "train_end_index": self.train_end_index
            if train_end_index is None
            else train_end_index,
        }


def _module_device_dtype(
    module: nn.Module,
) -> tuple[torch.device | None, torch.dtype | None]:
    """Return the device/dtype of a module without requiring it to own parameters."""

    first_tensor = next(iter(module.parameters()), None)
    if first_tensor is None:
        first_tensor = next(iter(module.buffers()), None)
    if first_tensor is None:
        return None, None
    dtype = first_tensor.dtype if first_tensor.is_floating_point() else None
    return first_tensor.device, dtype


def _extract_output_tensor(output: Any, attribute: str, operation: str) -> torch.Tensor:
    tensor = getattr(output, attribute, None)
    if tensor is None and isinstance(output, (tuple, list)) and output:
        tensor = output[0]
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"VAE {operation} output must expose a tensor as '.{attribute}'"
        )
    return tensor


class CelebALatentCodec(nn.Module):
    """Frozen wrapper around a diffusers ``AutoencoderKL``.

    Parameters
    ----------
    vae:
        An ``AutoencoderKL``-compatible module.  Supplying it explicitly keeps
        the class mockable and makes cached-latent training independent of
        diffusers.
    model_id:
        Identifier recorded in caches/checkpoints.  It is not downloaded by
        this constructor; use :meth:`from_pretrained` for that.
    scaling_factor:
        Multiplicative factor applied to posterior means.  If omitted, the
        VAE config value is used, with the Stable-Diffusion default as fallback.
    input_range:
        ``"zero_one"`` for this repository's CelebA transform, or
        ``"minus_one_one"`` when inputs are already normalized for the VAE.

    :meth:`encode` retains the historical deterministic posterior-mean
    behavior.  :meth:`encode_posterior_sample` is an explicit, reproducible
    posterior-draw path for matching the released Drifting Models recipe.
    """

    def __init__(
        self,
        vae: nn.Module,
        *,
        model_id: str = DEFAULT_VAE_MODEL_ID,
        model_files_sha256: Mapping[str, str] | None = None,
        scaling_factor: float | None = None,
        input_range: str = "zero_one",
        check_shapes: bool = True,
    ) -> None:
        super().__init__()
        if input_range not in {"zero_one", "minus_one_one"}:
            raise ValueError("input_range must be 'zero_one' or 'minus_one_one'")
        if scaling_factor is None:
            scaling_factor = getattr(
                getattr(vae, "config", None), "scaling_factor", None
            )
        if scaling_factor is None:
            scaling_factor = DEFAULT_VAE_SCALING_FACTOR
        if not math.isfinite(float(scaling_factor)) or float(scaling_factor) <= 0:
            raise ValueError("scaling_factor must be finite and positive")

        self.vae = vae
        self.model_id = str(model_id)
        self.model_files_sha256 = (
            None if model_files_sha256 is None else dict(model_files_sha256)
        )
        if self.model_files_sha256 is not None and not all(
            isinstance(name, str) and _valid_sha256(digest)
            for name, digest in self.model_files_sha256.items()
        ):
            raise ValueError(
                "model_files_sha256 must map relative paths to SHA-256 strings"
            )
        self.scaling_factor = float(scaling_factor)
        self.input_range = input_range
        self.check_shapes = bool(check_shapes)
        self._freeze_vae()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_VAE_MODEL_ID,
        *,
        scaling_factor: float | None = None,
        input_range: str = "zero_one",
        check_shapes: bool = True,
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype | None = None,
        **from_pretrained_kwargs: Any,
    ) -> "CelebALatentCodec":
        """Load a diffusers VAE while keeping diffusers an optional dependency."""

        try:
            from diffusers import AutoencoderKL
        except ModuleNotFoundError as exc:
            if exc.name != "diffusers":
                raise
            raise ImportError(
                "diffusers is required only to load/encode the VAE; install it "
                "or inject an AutoencoderKL-compatible module"
            ) from exc

        local_model_path = Path(model_id).resolve()
        model_files_sha256 = (
            directory_file_hashes(local_model_path)
            if local_model_path.is_dir()
            else None
        )
        if torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = torch_dtype
        vae = AutoencoderKL.from_pretrained(model_id, **from_pretrained_kwargs)
        if (
            model_files_sha256 is not None
            and directory_file_hashes(local_model_path) != model_files_sha256
        ):
            raise RuntimeError("local VAE files changed while loading the model")
        codec = cls(
            vae,
            model_id=model_id,
            model_files_sha256=model_files_sha256,
            scaling_factor=scaling_factor,
            input_range=input_range,
            check_shapes=check_shapes,
        )
        if device is not None:
            codec.to(device)
        return codec

    def _freeze_vae(self) -> None:
        self.vae.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True) -> "CelebALatentCodec":
        # A parent training module may call train() recursively.  The VAE is a
        # fixed coordinate chart and must remain in inference mode regardless.
        super().train(mode)
        self.vae.eval()
        return self

    def _move_to_vae(self, tensor: torch.Tensor) -> torch.Tensor:
        device, dtype = _module_device_dtype(self.vae)
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None and tensor.is_floating_point():
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs) if kwargs else tensor

    @staticmethod
    def _check_image_shape(images: torch.Tensor) -> None:
        if images.ndim != 4 or tuple(images.shape[1:]) != CELEBA_IMAGE_SHAPE:
            raise ValueError(
                f"expected CelebA images shaped [B, {CELEBA_IMAGE_SHAPE[0]}, "
                f"{CELEBA_IMAGE_SHAPE[1]}, {CELEBA_IMAGE_SHAPE[2]}], got {tuple(images.shape)}"
            )

    @staticmethod
    def _check_latent_shape(latents: torch.Tensor) -> None:
        if latents.ndim != 4 or tuple(latents.shape[1:]) != CELEBA_LATENT_SHAPE:
            raise ValueError(
                f"expected VAE latents shaped [B, {CELEBA_LATENT_SHAPE[0]}, "
                f"{CELEBA_LATENT_SHAPE[1]}, {CELEBA_LATENT_SHAPE[2]}], got {tuple(latents.shape)}"
            )

    def _encode_distribution(self, images: torch.Tensor) -> Any:
        self._check_image_shape(images)
        images = self._move_to_vae(images)
        vae_images = (
            images.mul(2.0).sub(1.0) if self.input_range == "zero_one" else images
        )
        encoded = self.vae.encode(vae_images)
        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is None:
            raise TypeError("VAE encode output must expose '.latent_dist'")
        return latent_dist

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode 64x64 RGB images to deterministic scaled ``[B, 4, 8, 8]`` means."""

        latent_dist = self._encode_distribution(images)
        posterior_mean = getattr(latent_dist, "mean", None)
        if not torch.is_tensor(posterior_mean):
            raise TypeError("VAE latent distribution must expose a tensor '.mean'")
        latents = posterior_mean * self.scaling_factor
        if self.check_shapes:
            self._check_latent_shape(latents)
        return latents

    @torch.no_grad()
    def encode_posterior_sample(
        self,
        images: torch.Tensor,
        *,
        seed: int = DEFAULT_POSTERIOR_SAMPLE_SEED,
        index_start: int = 0,
    ) -> torch.Tensor:
        """Encode one fixed posterior draw per absolute, contiguous image index.

        A separate CPU generator is deterministically derived for every image.
        Consequently, assigning a draw to an image does not depend on the
        loader batch size or worker count.  The explicit
        ``mean + std * epsilon`` operation is the one performed by
        ``latent_dist.sample`` in the official recipe, without relying on
        diffusers' untyped support for a list of generators.
        """

        _validate_posterior_sample_coordinates(seed, index_start)
        latent_dist = self._encode_distribution(images)
        posterior_mean = getattr(latent_dist, "mean", None)
        posterior_std = getattr(latent_dist, "std", None)
        if not torch.is_tensor(posterior_mean) or not torch.is_tensor(posterior_std):
            raise TypeError(
                "VAE latent distribution must expose tensor '.mean' and '.std'"
            )
        if posterior_std.shape != posterior_mean.shape:
            raise ValueError("VAE posterior mean and standard deviation shapes differ")
        generators = [
            torch.Generator(device="cpu").manual_seed(
                _posterior_sample_image_seed(seed, index_start + offset)
            )
            for offset in range(images.shape[0])
        ]
        noise = torch.cat(
            [
                torch.randn(
                    (1, *posterior_mean.shape[1:]),
                    generator=generator,
                    dtype=posterior_mean.dtype,
                    device="cpu",
                )
                for generator in generators
            ],
            dim=0,
        ).to(device=posterior_mean.device)
        posterior_sample = posterior_mean + posterior_std * noise
        latents = posterior_sample * self.scaling_factor
        if self.check_shapes:
            self._check_latent_shape(latents)
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
        """Decode scaled spatial latents back to RGB images in ``[0, 1]``."""

        if self.check_shapes:
            self._check_latent_shape(latents)
        latents = self._move_to_vae(latents)
        decoded = self.vae.decode(latents / self.scaling_factor)
        vae_images = _extract_output_tensor(decoded, "sample", "decode")
        images = vae_images.add(1.0).mul(0.5)
        if clamp:
            images = images.clamp(0.0, 1.0)
        if self.check_shapes:
            self._check_image_shape(images)
        return images

    def encode_flat(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images directly into the 2-D representation expected by ``flowmaps.objectives.wgf``."""

        return flatten_latents(self.encode(images))

    def decode_flat(
        self, flat_latents: torch.Tensor, *, clamp: bool = True
    ) -> torch.Tensor:
        """Decode flattened WGF particles to displayable images."""

        return self.decode(unflatten_latents(flat_latents), clamp=clamp)

    def cache_metadata(
        self,
        *,
        encoding: str = "posterior_mean",
        posterior_sample_seed: int = DEFAULT_POSTERIOR_SAMPLE_SEED,
        posterior_sample_index_start: int = 0,
    ) -> dict[str, Any]:
        if encoding not in LATENT_CACHE_ENCODINGS:
            raise ValueError(f"encoding must be one of {LATENT_CACHE_ENCODINGS}")
        metadata = {
            "model_id": self.model_id,
            "scaling_factor": self.scaling_factor,
            "input_range": self.input_range,
            "encoding": encoding,
        }
        if encoding == "posterior_sample":
            _validate_posterior_sample_coordinates(
                posterior_sample_seed,
                posterior_sample_index_start,
            )
            metadata.update(
                {
                    "posterior_sample_seed": posterior_sample_seed,
                    "posterior_sample_index_start": posterior_sample_index_start,
                    "posterior_sample_rng_scheme": POSTERIOR_SAMPLE_RNG_SCHEME,
                    "posterior_sample_indexing": POSTERIOR_SAMPLE_INDEXING,
                    "posterior_sample_horizontal_flip": False,
                }
            )
        if self.model_files_sha256 is not None:
            metadata["vae_files_sha256"] = dict(self.model_files_sha256)
        return metadata

    def assert_model_files_unchanged(self) -> None:
        """Refuse to attest a cache after its local VAE files changed."""

        if self.model_files_sha256 is None:
            return
        path = Path(self.model_id).resolve()
        if directory_file_hashes(path) != self.model_files_sha256:
            raise RuntimeError("local VAE files changed after the model was loaded")


def flatten_latents(latents: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, 4, 8, 8]`` latents to WGF particles ``[B, 256]``."""

    CelebALatentCodec._check_latent_shape(latents)
    return latents.reshape(latents.shape[0], CELEBA_LATENT_DIM)


def unflatten_latents(flat_latents: torch.Tensor) -> torch.Tensor:
    """Convert WGF particles ``[B, 256]`` back to spatial VAE latents."""

    if flat_latents.ndim != 2 or flat_latents.shape[1] != CELEBA_LATENT_DIM:
        raise ValueError(
            f"expected flattened latents shaped [B, {CELEBA_LATENT_DIM}], "
            f"got {tuple(flat_latents.shape)}"
        )
    return flat_latents.reshape(flat_latents.shape[0], *CELEBA_LATENT_SHAPE)


def _cache_payload(
    latents: torch.Tensor,
    *,
    codec: CelebALatentCodec,
    metadata: Mapping[str, Any] | None,
    encoding: str = "posterior_mean",
    posterior_sample_seed: int = DEFAULT_POSTERIOR_SAMPLE_SEED,
    posterior_sample_index_start: int = 0,
) -> dict[str, Any]:
    CelebALatentCodec._check_latent_shape(latents)
    reserved = {
        "format",
        "version",
        "model_id",
        "scaling_factor",
        "input_range",
        "encoding",
        "vae_files_sha256",
        "image_shape",
        "latent_shape",
        "count",
        "dtype",
        "posterior_sample_seed",
        "posterior_sample_index_start",
        "posterior_sample_rng_scheme",
        "posterior_sample_indexing",
        "posterior_sample_horizontal_flip",
    }
    extra = dict(metadata or {})
    overlap = reserved.intersection(extra)
    if overlap:
        raise ValueError(
            f"cache metadata cannot override reserved keys: {sorted(overlap)}"
        )
    if (
        encoding == "posterior_sample"
        and "start_index" in extra
        and extra["start_index"] != posterior_sample_index_start
    ):
        raise ValueError(
            "posterior_sample_index_start must equal metadata start_index "
            "for absolute per-image sampling"
        )
    cache_metadata = {
        "format": LATENT_CACHE_FORMAT,
        "version": LATENT_CACHE_VERSION,
        **codec.cache_metadata(
            encoding=encoding,
            posterior_sample_seed=posterior_sample_seed,
            posterior_sample_index_start=posterior_sample_index_start,
        ),
        "image_shape": list(CELEBA_IMAGE_SHAPE),
        "latent_shape": list(CELEBA_LATENT_SHAPE),
        "count": latents.shape[0],
        "dtype": str(latents.dtype),
        **extra,
    }
    return {"latents": latents.detach().cpu().contiguous(), "metadata": cache_metadata}


def save_latent_cache(
    path: str | Path,
    latents: torch.Tensor,
    *,
    codec: CelebALatentCodec,
    metadata: Mapping[str, Any] | None = None,
    encoding: str = "posterior_mean",
    posterior_sample_seed: int = DEFAULT_POSTERIOR_SAMPLE_SEED,
    posterior_sample_index_start: int = 0,
    overwrite_existing: bool = False,
    expected_parent_identity: tuple[int, int] | None = None,
    staging_name: str | None = None,
) -> Path:
    """Atomically create a spatial latent cache through one held parent fd.

    The payload is serialized to an exclusively-created private leaf, fsynced,
    and published with ``RENAME_NOREPLACE``.  The official cache route supplies
    the identity of its already-created partial bundle and never enables
    ``overwrite_existing``.
    """

    path = Path(path).absolute()
    if not path.name or path.name in {".", ".."} or "/" in path.name:
        raise ValueError("latent-cache destination name is malformed")
    codec.assert_model_files_unchanged()
    payload = _cache_payload(
        latents,
        codec=codec,
        metadata=metadata,
        encoding=encoding,
        posterior_sample_seed=posterior_sample_seed,
        posterior_sample_index_start=posterior_sample_index_start,
    )
    parent_metadata = official_runtime.secure_directory_metadata(path.parent)
    parent_identity = (parent_metadata["device"], parent_metadata["inode"])
    if (
        expected_parent_identity is not None
        and parent_identity != expected_parent_identity
    ):
        raise RuntimeError(
            "latent-cache parent identity differs from its caller's partial"
        )
    if staging_name is None:
        staging_name = f".{path.name}.staging-{secrets.token_hex(32)}"
    if (
        type(staging_name) is not str
        or not staging_name.startswith(f".{path.name}.staging-")
        or "/" in staging_name
        or "\x00" in staging_name
    ):
        raise ValueError("latent-cache staging name is malformed")
    with official_runtime.secure_directory_fd(
        path.parent, expected_identity=parent_identity
    ) as parent_descriptor:
        if overwrite_existing:
            try:
                prior = os.stat(
                    path.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(prior.st_mode):
                    raise ValueError("latent-cache overwrite target is not regular")
                official_runtime.secure_unlink_regular_at(
                    parent_descriptor,
                    path.name,
                    expected_identity=(prior.st_dev, prior.st_ino),
                )
        descriptor = -1
        staging_identity: tuple[int, int] | None = None
        published = False
        try:
            descriptor = os.open(
                staging_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
            created = os.fstat(descriptor)
            if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
                raise RuntimeError(
                    "latent-cache staging leaf is not a private regular file"
                )
            staging_identity = (created.st_dev, created.st_ino)
            with os.fdopen(os.dup(descriptor), "wb") as output:
                torch.save(payload, output)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(descriptor)
            if (
                not stat.S_ISREG(after.st_mode)
                or (after.st_dev, after.st_ino) != staging_identity
                or after.st_size <= 0
                or after.st_nlink != 1
            ):
                raise RuntimeError(
                    "latent-cache staging identity changed during serialization"
                )
            os.fsync(descriptor)
            official_runtime.secure_publish_regular_noreplace_at(
                parent_descriptor,
                staging_name,
                path.name,
                expected_source_identity=staging_identity,
            )
            published = True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if not published and staging_identity is not None:
                try:
                    official_runtime.secure_unlink_regular_at(
                        parent_descriptor,
                        staging_name,
                        expected_identity=staging_identity,
                    )
                except FileNotFoundError:
                    pass
    return path


def _validate_cache_encoding_metadata(metadata: Mapping[str, Any]) -> str:
    encoding = metadata.get("encoding")
    if encoding not in LATENT_CACHE_ENCODINGS:
        raise ValueError(
            f"latent cache has unsupported encoding {encoding!r}; "
            f"expected one of {LATENT_CACHE_ENCODINGS}"
        )

    sample_keys = {
        "posterior_sample_seed",
        "posterior_sample_index_start",
        "posterior_sample_rng_scheme",
        "posterior_sample_indexing",
        "posterior_sample_horizontal_flip",
    }
    if encoding == "posterior_mean":
        unexpected = sorted(sample_keys.intersection(metadata))
        if unexpected:
            raise ValueError(
                "posterior-mean cache unexpectedly contains posterior-sample metadata "
                f"{unexpected}"
            )
        return encoding

    missing = sorted(sample_keys.difference(metadata))
    if missing:
        raise ValueError(f"posterior-sample cache lacks required metadata {missing}")
    seed = metadata["posterior_sample_seed"]
    index_start = metadata["posterior_sample_index_start"]
    _validate_posterior_sample_coordinates(seed, index_start)
    if metadata["posterior_sample_rng_scheme"] != POSTERIOR_SAMPLE_RNG_SCHEME:
        raise ValueError("posterior-sample cache uses an unsupported RNG scheme")
    if metadata["posterior_sample_indexing"] != POSTERIOR_SAMPLE_INDEXING:
        raise ValueError(
            "posterior-sample cache uses an unsupported image-indexing scheme"
        )
    if metadata["posterior_sample_horizontal_flip"] is not False:
        raise ValueError("posterior-sample cache must not use horizontal flips")
    if "start_index" in metadata and metadata["start_index"] != index_start:
        raise ValueError(
            "posterior-sample cache index start disagrees with its dataset start_index"
        )
    return encoding


def load_latent_cache(
    path: str | Path | Any,
    *,
    expected_file_sha256: str | None = None,
    expected_model_id: str | None = None,
    expected_scaling_factor: float | None = None,
    expected_input_range: str | None = None,
    expected_encoding: str | None = None,
    expected_posterior_sample_seed: int | None = None,
    expected_posterior_sample_index_start: int | None = None,
    map_location: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and validate a cache, refusing silently incompatible VAE coordinates."""

    if hasattr(path, "read"):
        if expected_file_sha256 is not None:
            raise ValueError(
                "expected_file_sha256 requires a filesystem path so its binding can be sealed"
            )
        payload = torch.load(path, map_location=map_location, weights_only=True)
    else:
        with official_runtime.secure_regular_fd(
            Path(path).absolute(), expected_sha256=expected_file_sha256
        ) as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                payload = torch.load(
                    handle, map_location=map_location, weights_only=True
                )
    if not isinstance(payload, dict) or set(payload) != {"latents", "metadata"}:
        raise ValueError("not a driftmodel CelebA latent cache")
    latents = payload["latents"]
    metadata = payload["metadata"]
    if not torch.is_tensor(latents) or not isinstance(metadata, dict):
        raise ValueError("malformed CelebA latent cache")
    if metadata.get("format") != LATENT_CACHE_FORMAT:
        raise ValueError("unknown CelebA latent cache format")
    if metadata.get("version") != LATENT_CACHE_VERSION:
        raise ValueError(
            f"unsupported CelebA latent cache version: {metadata.get('version')}"
        )
    CelebALatentCodec._check_latent_shape(latents)
    if metadata.get("count") != latents.shape[0]:
        raise ValueError("latent cache count does not match its tensor")
    if metadata.get("dtype") != str(latents.dtype):
        raise ValueError("latent cache dtype metadata does not match its tensor")
    if list(metadata.get("latent_shape", ())) != list(CELEBA_LATENT_SHAPE):
        raise ValueError("latent cache has an incompatible spatial shape")
    encoding = _validate_cache_encoding_metadata(metadata)
    vae_hashes = metadata.get("vae_files_sha256")
    if vae_hashes is not None and (
        not isinstance(vae_hashes, Mapping)
        or not vae_hashes
        or not all(
            isinstance(name, str) and _valid_sha256(digest)
            for name, digest in vae_hashes.items()
        )
    ):
        raise ValueError("latent cache has malformed VAE file hashes")
    if expected_model_id is not None and metadata.get("model_id") != expected_model_id:
        raise ValueError(
            f"latent cache VAE mismatch: expected {expected_model_id!r}, "
            f"found {metadata.get('model_id')!r}"
        )
    if (
        expected_input_range is not None
        and metadata.get("input_range") != expected_input_range
    ):
        raise ValueError(
            f"latent cache input range mismatch: expected {expected_input_range!r}, "
            f"found {metadata.get('input_range')!r}"
        )
    if expected_encoding is not None:
        if expected_encoding not in LATENT_CACHE_ENCODINGS:
            raise ValueError(
                f"expected_encoding must be one of {LATENT_CACHE_ENCODINGS}"
            )
        if encoding != expected_encoding:
            raise ValueError(
                f"latent cache encoding mismatch: expected {expected_encoding!r}, "
                f"found {encoding!r}"
            )
    if expected_posterior_sample_seed is not None:
        if metadata.get("posterior_sample_seed") != expected_posterior_sample_seed:
            raise ValueError(
                "latent cache posterior-sample seed mismatch: "
                f"expected {expected_posterior_sample_seed}, "
                f"found {metadata.get('posterior_sample_seed')}"
            )
    if expected_posterior_sample_index_start is not None:
        if (
            metadata.get("posterior_sample_index_start")
            != expected_posterior_sample_index_start
        ):
            raise ValueError(
                "latent cache posterior-sample index start mismatch: "
                f"expected {expected_posterior_sample_index_start}, "
                f"found {metadata.get('posterior_sample_index_start')}"
            )
    if expected_scaling_factor is not None and not math.isclose(
        float(metadata.get("scaling_factor", float("nan"))),
        float(expected_scaling_factor),
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "latent cache scaling mismatch: "
            f"expected {expected_scaling_factor}, found {metadata.get('scaling_factor')}"
        )
    return latents, metadata


def _images_from_batch(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("images", "image", "pixel_values"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
        raise TypeError("mapping batches must contain tensor images/image/pixel_values")
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    raise TypeError(
        "loader batches must be image tensors, mappings, or sequences starting with images"
    )


@torch.no_grad()
def encode_loader_to_cache(
    codec: CelebALatentCodec,
    loader: Iterable[Any],
    path: str | Path,
    *,
    overwrite: bool = False,
    max_items: int | None = None,
    cache_dtype: torch.dtype = torch.float32,
    metadata: Mapping[str, Any] | None = None,
    encoding: str = "posterior_mean",
    posterior_sample_seed: int = DEFAULT_POSTERIOR_SAMPLE_SEED,
    posterior_sample_index_start: int = 0,
    expected_parent_identity: tuple[int, int] | None = None,
    staging_name: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode a loader once, or reuse a compatible existing cache.

    Only image tensors are cached; CelebA attributes are intentionally ignored
    because the unconditional WGF objective does not consume them.  The default
    ``posterior_mean`` path is unchanged.  ``posterior_sample`` assigns one
    fixed draw to every absolute image index, independently of batching.
    That mode requires the iterable to yield a contiguous, ordered image slice
    beginning at ``posterior_sample_index_start``; the cache CLI enforces this
    with its non-shuffled, non-dropping loader.
    Returned tensors live on CPU and can be wrapped in
    ``TensorDataset``/``DataLoader``.
    """

    path = Path(path)
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive when provided")
    encoding_metadata = codec.cache_metadata(
        encoding=encoding,
        posterior_sample_seed=posterior_sample_seed,
        posterior_sample_index_start=posterior_sample_index_start,
    )
    if (
        encoding == "posterior_sample"
        and "start_index" in (metadata or {})
        and (metadata or {})["start_index"] != posterior_sample_index_start
    ):
        raise ValueError(
            "posterior_sample_index_start must equal metadata start_index "
            "for absolute per-image sampling"
        )
    if path.exists() and not overwrite:
        cached_latents, cached_metadata = load_latent_cache(
            path,
            expected_model_id=codec.model_id,
            expected_scaling_factor=codec.scaling_factor,
            expected_input_range=codec.input_range,
            expected_encoding=encoding,
            expected_posterior_sample_seed=(
                posterior_sample_seed if encoding == "posterior_sample" else None
            ),
            expected_posterior_sample_index_start=(
                posterior_sample_index_start if encoding == "posterior_sample" else None
            ),
        )
        expected_metadata = {**encoding_metadata, **dict(metadata or {})}
        mismatches = sorted(
            key
            for key, expected in expected_metadata.items()
            if cached_metadata.get(key) != expected
        )
        if cached_metadata.get("dtype") != str(cache_dtype):
            mismatches.append("dtype")
        if max_items is not None and cached_latents.shape[0] > max_items:
            mismatches.append("max_items")
        if mismatches:
            raise ValueError(
                "existing latent cache differs from requested settings for "
                f"{sorted(set(mismatches))}; pass overwrite=True to rebuild it"
            )
        return cached_latents, cached_metadata

    chunks: list[torch.Tensor] = []
    count = 0
    for batch in loader:
        images = _images_from_batch(batch)
        if max_items is not None:
            remaining = max_items - count
            if remaining <= 0:
                break
            images = images[:remaining]
        if images.shape[0] == 0:
            continue
        if encoding == "posterior_sample":
            chunk = codec.encode_posterior_sample(
                images,
                seed=posterior_sample_seed,
                index_start=posterior_sample_index_start + count,
            )
        else:
            chunk = codec.encode(images)
        chunk = chunk.to(device="cpu", dtype=cache_dtype)
        chunks.append(chunk)
        count += chunk.shape[0]
        if max_items is not None and count >= max_items:
            break
    if not chunks:
        raise ValueError("cannot build a latent cache from an empty loader")
    latents = torch.cat(chunks, dim=0)
    save_latent_cache(
        path,
        latents,
        codec=codec,
        metadata=metadata,
        encoding=encoding,
        posterior_sample_seed=posterior_sample_seed,
        posterior_sample_index_start=posterior_sample_index_start,
        overwrite_existing=overwrite,
        expected_parent_identity=expected_parent_identity,
        staging_name=staging_name,
    )
    saved_sha256 = official_runtime.secure_sha256_file(path.absolute())
    return load_latent_cache(
        path,
        expected_file_sha256=saved_sha256,
        expected_model_id=codec.model_id,
        expected_scaling_factor=codec.scaling_factor,
        expected_input_range=codec.input_range,
        expected_encoding=encoding,
        expected_posterior_sample_seed=(
            posterior_sample_seed if encoding == "posterior_sample" else None
        ),
        expected_posterior_sample_index_start=(
            posterior_sample_index_start if encoding == "posterior_sample" else None
        ),
    )


def _time_column(
    value: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(1, 1).expand(batch_size, 1)
    if value.numel() != batch_size:
        raise ValueError(
            f"time must be scalar or have {batch_size} elements, got {tuple(value.shape)}"
        )
    return value.reshape(batch_size, 1)


def flow_map_endpoint_weight(
    t: torch.Tensor | float,
    s: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """MNIST endpoint interpolation weight ``clamp((s-t)/(1-t), 0, 1)``.

    The removable ``0/0`` case at ``t=s=1`` is defined as zero, preserving
    ``Phi(x, t, t) = x``.  No epsilon is added for ``t < 1``: consequently a
    transition from any such ``t`` to exactly 1 has weight exactly 1 and fully
    suppresses the explicit source term.
    """

    start = _time_column(t, batch_size=batch_size, device=device, dtype=dtype)
    end = _time_column(s, batch_size=batch_size, device=device, dtype=dtype)
    remaining = 1.0 - start
    safe_remaining = torch.where(remaining == 0, torch.ones_like(remaining), remaining)
    raw_weight = (end - start) / safe_remaining
    # ``Tensor.clamp(min=0)`` has a zero autograd derivative exactly at zero.
    # WGF's diagonal JVP is evaluated at s=t and needs the interior/right
    # derivative 1/(1-t), so express the same value projection with strict
    # inequalities.  Values are still exactly clamped outside [0, 1].
    weight = torch.where(raw_weight < 0.0, torch.zeros_like(raw_weight), raw_weight)
    return torch.where(weight > 1.0, torch.ones_like(weight), weight)


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("sinusoidal embedding dimension must be at least 4")
        half_dim = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        self.dim = dim
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = (
            time[:, None]
            * self.frequencies[None].to(dtype=time.dtype)
            * (2.0 * math.pi)
        )
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConditionedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(condition_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(x)


class TransitionConditionedLatentUNet(nn.Module):
    """Compact two-time U-Net endpoint map for 4x8x8 CelebA latents.

    In historical ``absolute`` mode the network predicts an unrestricted
    latent endpoint ``g_theta`` and uses the MNIST interpolation invariant::

        w(t, s) = clamp((s - t) / (1 - t), 0, 1)
        Phi_theta(x, t, s) = (1 - w) x + w g_theta(x, t, s)

    Optional ``residual`` mode instead sets ``g_theta=x+r_theta`` and evaluates
    the algebraically equivalent ``Phi=x+w*r_theta``. A zero-initialized output
    is then bitwise identity rather than the historical zero endpoint.

    In absolute mode the diagonal is an exact identity, while every map to ``s=1``
    has zero explicit coefficient on its source.  Along a partition, the
    retained-source coefficients telescope because
    ``1-w(t,s)=(1-s)/(1-t)``.  ``g_theta`` has no sigmoid/tanh because scaled
    AutoencoderKL latents are Euclidean and unbounded.

    Both flattened ``[B, 256]`` particles (for :mod:`flowmaps.objectives.wgf`) and spatial
    ``[B, 4, 8, 8]`` tensors are accepted; output layout matches input layout.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        time_embedding_dim: int = 32,
        condition_dim: int | None = None,
        zero_init_endpoint: bool = True,
        endpoint_parameterization: str = "absolute",
    ) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        condition_dim = 4 * base_channels if condition_dim is None else condition_dim
        if condition_dim < 4:
            raise ValueError("condition_dim must be at least 4")
        if endpoint_parameterization not in {"absolute", "residual"}:
            raise ValueError(
                "endpoint_parameterization must be 'absolute' or 'residual'"
            )
        self.endpoint_parameterization = endpoint_parameterization

        self.time_embedding = _SinusoidalEmbedding(time_embedding_dim)
        self.transition_mlp = nn.Sequential(
            nn.Linear(2 * time_embedding_dim + 1, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        high_channels = 2 * base_channels
        self.input_conv = nn.Conv2d(
            CELEBA_LATENT_SHAPE[0], base_channels, kernel_size=3, padding=1
        )
        self.down_block = _ConditionedResidualBlock(
            base_channels, base_channels, condition_dim
        )
        self.downsample = nn.Conv2d(
            base_channels, high_channels, kernel_size=4, stride=2, padding=1
        )
        self.middle_block1 = _ConditionedResidualBlock(
            high_channels, high_channels, condition_dim
        )
        self.middle_block2 = _ConditionedResidualBlock(
            high_channels, high_channels, condition_dim
        )
        self.upsample = nn.ConvTranspose2d(
            high_channels,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block = _ConditionedResidualBlock(
            2 * base_channels, base_channels, condition_dim
        )
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output_conv = nn.Conv2d(
            base_channels, CELEBA_LATENT_SHAPE[0], kernel_size=3, padding=1
        )
        if zero_init_endpoint:
            nn.init.zeros_(self.output_conv.weight)
            nn.init.zeros_(self.output_conv.bias)

    @staticmethod
    def _spatial_input(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return unflatten_latents(x), True
        CelebALatentCodec._check_latent_shape(x)
        return x, False

    def _transition_condition(
        self,
        t: torch.Tensor | float,
        s: torch.Tensor | float,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = like.shape[0]
        start = _time_column(
            t, batch_size=batch_size, device=like.device, dtype=like.dtype
        )
        end = _time_column(
            s, batch_size=batch_size, device=like.device, dtype=like.dtype
        )
        weight = flow_map_endpoint_weight(
            start,
            end,
            batch_size=batch_size,
            device=like.device,
            dtype=like.dtype,
        )
        condition = self.transition_mlp(
            torch.cat(
                [
                    self.time_embedding(start[:, 0]),
                    self.time_embedding(end[:, 0]),
                    weight,
                ],
                dim=1,
            )
        )
        return condition, weight

    def endpoint(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        s: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return the learned unbounded endpoint ``g_theta`` without interpolation."""

        spatial, was_flat = self._spatial_input(x)
        condition, _ = self._transition_condition(t, s, spatial)
        first_skip = self.down_block(self.input_conv(spatial), condition)
        hidden = self.downsample(first_skip)
        hidden = self.middle_block1(hidden, condition)
        hidden = self.middle_block2(hidden, condition)
        hidden = self.upsample(hidden)
        if hidden.shape[-2:] != first_skip.shape[-2:]:
            hidden = F.interpolate(hidden, size=first_skip.shape[-2:], mode="nearest")
        hidden = self.up_block(torch.cat([hidden, first_skip], dim=1), condition)
        endpoint = self.output_conv(F.silu(self.output_norm(hidden)))
        if self.endpoint_parameterization == "residual":
            endpoint = spatial + endpoint
        return flatten_latents(endpoint) if was_flat else endpoint

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        s: torch.Tensor | float,
    ) -> torch.Tensor:
        spatial, was_flat = self._spatial_input(x)
        condition, weight = self._transition_condition(t, s, spatial)

        first_skip = self.down_block(self.input_conv(spatial), condition)
        hidden = self.downsample(first_skip)
        hidden = self.middle_block1(hidden, condition)
        hidden = self.middle_block2(hidden, condition)
        hidden = self.upsample(hidden)
        if hidden.shape[-2:] != first_skip.shape[-2:]:
            hidden = F.interpolate(hidden, size=first_skip.shape[-2:], mode="nearest")
        hidden = self.up_block(torch.cat([hidden, first_skip], dim=1), condition)
        endpoint = self.output_conv(F.silu(self.output_norm(hidden)))
        if self.endpoint_parameterization == "residual":
            # Keep zero-residual initialization bitwise equal to the identity;
            # the algebraically equivalent endpoint interpolation can round.
            output = spatial + weight[:, :, None, None] * endpoint
        else:
            output = (1.0 - weight[:, :, None, None]) * spatial + weight[
                :, :, None, None
            ] * endpoint
        return flatten_latents(output) if was_flat else output


# Explicit experiment-facing alias; both names describe the same WGF map.
CelebALatentFlowMap = TransitionConditionedLatentUNet


__all__ = [
    "CELEBA_IMAGE_SHAPE",
    "CELEBA_LATENT_DIM",
    "CELEBA_LATENT_SHAPE",
    "DEFAULT_VAE_MODEL_ID",
    "DEFAULT_VAE_SCALING_FACTOR",
    "DEFAULT_POSTERIOR_SAMPLE_SEED",
    "LATENT_CACHE_ENCODINGS",
    "LATENT_PRIOR_FILENAME",
    "LATENT_PRIOR_FORMAT",
    "LATENT_PRIOR_VERSION",
    "POSTERIOR_SAMPLE_INDEXING",
    "POSTERIOR_SAMPLE_RNG_SCHEME",
    "CelebALatentCodec",
    "CelebALatentFlowMap",
    "LatentAffineNormalization",
    "LatentGaussianPrior",
    "LatentGaussianSampler",
    "TransitionConditionedLatentUNet",
    "directory_file_hashes",
    "encode_loader_to_cache",
    "flatten_latents",
    "flow_map_endpoint_weight",
    "load_latent_cache",
    "save_latent_cache",
    "sha256_file",
    "sha256_tensor",
    "unflatten_latents",
]
