"""Frozen ConvNeXt-V2 features for a future CelebA-64 pixel-GMD branch.

This module is deliberately independent from the currently running CelebA
pixel-GMD trainer.  It implements the 13 ConvNeXt feature families exported by
``models/convnext.py`` in the official Drifting release at commit
``accd0cf09c33b70892d33941d2a287ca86cb92e1``.

Public image tensors use the local ``[B, 3, 64, 64]`` NCHW convention.  The
official path maps model-space RGB from ``[-1, 1]`` to unclipped ``[0, 1]``,
applies ImageNet normalization, and bilinearly resizes to 224 pixels.  Frozen
parameters do not disable the input Jacobian: generated images remain in the
autograd graph, while real supports are extracted under ``torch.no_grad``.

No function in this module downloads a model.  The production loader accepts
only a local Hugging Face snapshot and verifies the pinned config and
``model.safetensors`` before importing/loading Transformers.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


OFFICIAL_RELEASE_COMMIT = "accd0cf09c33b70892d33941d2a287ca86cb92e1"

HF_MODEL_ID = "facebook/convnextv2-base-22k-224"
HF_MODEL_REVISION = "758ff0922dc09136abb55774e7f8b1e1bd0dc344"
HF_CONFIG_FILENAME = "config.json"
HF_CONFIG_SIZE_BYTES = 69_727
HF_CONFIG_SHA256 = "dac818f34972086f8171c444adac7752e13466b201eee2e248768597aa114732"
HF_WEIGHTS_FILENAME = "model.safetensors"
HF_WEIGHTS_SIZE_BYTES = 354_917_376
HF_WEIGHTS_SHA256 = "8937dc20b1d7a63ea628752315a89b9a4fc0f58bf78b3299041e99c0ad3f93bb"

CONVNEXT_BACKBONE_PARAMETER_COUNT = 87_692_800
CONVNEXT_GRN_COUNT = 36
OFFICIAL_GRN_EPS = 1e-6
CELEBA_RGB_SHAPE = (3, 64, 64)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STAGE_NORMALIZATION_EPS = 1e-3
SAFE_STD_EPS = 1e-6

# This is deliberately described as a local PyTorch adaptation of the official
# JAX feature extractor.  It is not called a Transformers-5.3 reproduction:
# both Transformers 5.3.0 and the attested local checkout omit the epsilon
# *inside* the GRN square root, unlike the official Drifting JAX source.
LOCAL_RUNTIME_ADAPTATION = "drifting-jax-convnextv2-base-pytorch-adaptation-v2"
EXPECTED_TRANSFORMERS_VERSION = "5.15.0.dev0"
EXPECTED_TORCH_VERSION = "2.9.1+rocm6.3"
EXPECTED_HIP_VERSION = "6.3.42134-a9a80e791"
EXPECTED_TRANSFORMERS_SOURCE_SHA256 = {
    "configuration_convnextv2.py": (
        "597d53a0143ba92163de0977a101ade4618db1f440058526f810368077cb8b73"
    ),
    "modeling_convnextv2.py": (
        "a8d9a5d91d2edc81c614088d83b779b04447e2713d13759c7cf9651238ac988b"
    ),
    "activations.py": (
        "4b1a469c24be4e4c6320536ad55a7337d24de04c3ecc71597fbf0267f1560612"
    ),
}

# Keep both the ordering and the upstream ``convenxt`` spelling.  The official
# loss sums dictionaries in insertion order and persisted metric names include
# this typo.
CONVNEXT_FAMILY_SPECS: tuple[tuple[str, int, int], ...] = (
    ("convenxt_stage_0_mean", 1, 128),
    ("convenxt_stage_0_std", 1, 128),
    ("convenxt_stage_1", 28 * 28, 256),
    ("convenxt_stage_1_mean", 1, 256),
    ("convenxt_stage_1_std", 1, 256),
    ("convenxt_stage_2", 14 * 14, 512),
    ("convenxt_stage_2_mean", 1, 512),
    ("convenxt_stage_2_std", 1, 512),
    ("convenxt_stage_3", 7 * 7, 1024),
    ("convenxt_stage_3_mean", 1, 1024),
    ("convenxt_stage_3_std", 1, 1024),
    ("global_mean", 1, 1024),
    ("global_std", 1, 1024),
)
CONVNEXT_FAMILY_NAMES = tuple(spec[0] for spec in CONVNEXT_FAMILY_SPECS)
_STAGE_SPECS = ((128, 56), (256, 28), (512, 14), (1024, 7))


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 without materializing a large weight file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_images(images: Tensor, label: str = "images") -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != CELEBA_RGB_SHAPE:
        raise ValueError(f"{label} must have shape [B,3,64,64]")
    if images.shape[0] < 1 or not images.is_floating_point():
        raise ValueError(f"{label} must be a non-empty floating tensor")


def preprocess_rgb64_for_convnext(images: Tensor) -> Tensor:
    """Apply the release's differentiable RGB-64 -> ConvNeXt input path.

    Normalization intentionally precedes resize, matching
    ``build_activation_function`` followed by ``ConvNextV2.get_activations``
    in the release.  There is no clipping: generated training pixels are
    allowed to leave ``[-1, 1]``.
    """

    _validate_images(images)
    # The official input activation applies the affine in the input dtype.
    # Promote only afterwards so BF16 follows ``((x_bf16 + 1) / 2).float()``
    # rather than the observably different ``(x.float() + 1) / 2`` path.
    one = images.new_tensor(1.0)
    two = images.new_tensor(2.0)
    values = ((images + one) / two).float()
    mean = values.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = values.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    values = (values - mean) / std
    return F.interpolate(
        values,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )


def normalize_convnext_stage(feature_map: Tensor) -> Tensor:
    """Normalize channels independently at each spatial location.

    JAX's default standard deviation is the population statistic.  Computing
    the variance explicitly in FP32 avoids PyTorch's correction-version
    differences and matches the release's ``std + 1e-3`` denominator.
    """

    if feature_map.ndim != 4 or not feature_map.is_floating_point():
        raise ValueError("ConvNeXt stage features must be floating [B,C,H,W]")
    old_dtype = feature_map.dtype
    values = feature_map.float()
    mean = values.mean(dim=1, keepdim=True)
    variance = (values - mean).square().mean(dim=1, keepdim=True).clamp_min(0.0)
    normalized = (values - mean) / (variance.sqrt() + STAGE_NORMALIZATION_EPS)
    return normalized.to(dtype=old_dtype)


def safe_std(values: Tensor, dim: int | tuple[int, ...]) -> Tensor:
    """Release-compatible population standard deviation accumulated in FP32."""

    values_f32 = values.float()
    mean = values_f32.mean(dim=dim, keepdim=True)
    variance = (values_f32 - mean).square().mean(dim=dim).clamp_min(0.0)
    return (variance + SAFE_STD_EPS).sqrt()


def _tokens(feature_map: Tensor) -> Tensor:
    return feature_map.flatten(2).transpose(1, 2).contiguous()


class OfficialJaxConvNextV2GRN(nn.Module):
    """PyTorch form of the official Drifting JAX ``ConvNextGRN``.

    Hugging Face computes ``sqrt(sum(x**2))``.  The released JAX feature model
    instead computes ``sqrt(sum(x**2) + 1e-6)``.  That difference matters at
    and near zero and changes its Jacobian, so the HF module is replaced after
    strict weight loading.  The original Parameter objects are retained.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter) -> None:
        super().__init__()
        if not isinstance(weight, nn.Parameter) or not isinstance(bias, nn.Parameter):
            raise TypeError("GRN weight and bias must be torch Parameters")
        if weight.shape != bias.shape or weight.ndim != 4:
            raise ValueError("GRN weight and bias must have identical rank-four shapes")
        self.weight = weight
        self.bias = bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 4 or not hidden_states.is_floating_point():
            raise ValueError("GRN input must be a floating NHWC rank-four tensor")
        old_dtype = hidden_states.dtype
        norm = hidden_states.square().sum(dim=(1, 2), keepdim=True)
        eps = norm.new_tensor(OFFICIAL_GRN_EPS)
        global_features = torch.sqrt(norm + eps)
        norm_features = global_features / (
            global_features.mean(dim=-1, keepdim=True) + eps
        )
        output = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        return output.to(dtype=old_dtype)


def replace_hf_grn_with_official_jax(
    backbone: nn.Module,
    hf_grn_type: type[nn.Module],
    *,
    expected_count: int = CONVNEXT_GRN_COUNT,
) -> int:
    """Replace every HF GRN while preserving its exact weight/bias objects."""

    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone must be a torch.nn.Module")
    if not isinstance(hf_grn_type, type) or not issubclass(hf_grn_type, nn.Module):
        raise TypeError("hf_grn_type must be a torch.nn.Module class")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")

    found = sum(isinstance(module, hf_grn_type) for module in backbone.modules())
    if found != expected_count:
        raise RuntimeError(f"found {found} HF GRNs, expected exactly {expected_count}")

    replaced = 0

    def recurse(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, hf_grn_type):
                if not hasattr(child, "weight") or not hasattr(child, "bias"):
                    raise RuntimeError("HF GRN lacks weight or bias")
                replacement = OfficialJaxConvNextV2GRN(child.weight, child.bias)
                replacement.train(child.training)
                parent._modules[name] = replacement
                replaced += 1
            else:
                recurse(child)

    recurse(backbone)
    if replaced != expected_count:
        raise RuntimeError(f"replaced {replaced} HF GRNs, expected {expected_count}")
    if any(isinstance(module, hf_grn_type) for module in backbone.modules()):
        raise RuntimeError("an unpatched HF GRN remains in the backbone")
    if sum(
        isinstance(module, OfficialJaxConvNextV2GRN)
        for module in backbone.modules()
    ) != expected_count:
        raise RuntimeError("official-JAX GRN replacement count is inconsistent")
    return replaced


class CelebA64ConvNextV2Features(nn.Module):
    """Export the exact 13 frozen ConvNeXt-V2 Base drift feature families.

    ``backbone`` must expose the Hugging Face ConvNeXt-V2 backbone interface:
    ``embeddings``, four ``encoder.stages``, and the final ``layernorm``.  The
    injectable interface lets unit tests exercise every transformation without
    accessing pretrained weights.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        model_identity: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        if not hasattr(backbone, "embeddings") or not hasattr(backbone, "encoder"):
            raise ValueError("backbone lacks ConvNeXt embeddings or encoder")
        stages = getattr(backbone.encoder, "stages", None)
        if stages is None or len(stages) != 4:
            raise ValueError("backbone must expose exactly four ConvNeXt stages")
        if not hasattr(backbone, "layernorm"):
            raise ValueError("backbone lacks the final ConvNeXt layernorm")

        self.backbone = backbone
        self._model_identity = dict(model_identity or {})
        self.backbone.requires_grad_(False)
        super().train(False)

    @property
    def model_identity(self) -> dict[str, Any]:
        return dict(self._model_identity)

    def train(self, mode: bool = True) -> "CelebA64ConvNextV2Features":
        """Remain in eval mode even if a parent training module calls ``train``."""

        del mode
        super().train(False)
        return self

    def _assert_frozen_eval(self) -> None:
        if self.training or self.backbone.training:
            raise RuntimeError("ConvNeXt feature backbone must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("ConvNeXt feature parameters must remain frozen")

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        self._assert_frozen_eval()
        _validate_images(images)

        # Pixel-B leaves ConvNeXt in FP32 (convnext_bf16 defaults to false).
        # Disabling an outer trainer autocast is therefore part of this exact
        # adapter, while the cast preserves the gradient to a BF16 input.
        with torch.autocast(device_type=images.device.type, enabled=False):
            hidden = preprocess_rgb64_for_convnext(images)
            hidden = self.backbone.embeddings(hidden)
            output: dict[str, Tensor] = {}
            final_normalized: Tensor | None = None

            for stage_index, (stage, (channels, spatial)) in enumerate(
                zip(self.backbone.encoder.stages, _STAGE_SPECS)
            ):
                hidden = stage(hidden)
                expected = (images.shape[0], channels, spatial, spatial)
                if tuple(hidden.shape) != expected:
                    raise RuntimeError(
                        f"ConvNeXt stage {stage_index} shape {tuple(hidden.shape)} "
                        f"does not match {expected}"
                    )
                normalized = normalize_convnext_stage(hidden)
                if stage_index > 0:
                    output[f"convenxt_stage_{stage_index}"] = _tokens(normalized)
                output[f"convenxt_stage_{stage_index}_mean"] = normalized.mean(
                    dim=(-2, -1)
                ).unsqueeze(1)
                output[f"convenxt_stage_{stage_index}_std"] = safe_std(
                    normalized, (-2, -1)
                ).unsqueeze(1)
                final_normalized = normalized

            assert final_normalized is not None
            output["global_mean"] = self.backbone.layernorm(
                hidden.mean(dim=(-2, -1))
            ).unsqueeze(1)
            output["global_std"] = safe_std(_tokens(final_normalized), 1).unsqueeze(1)

        if tuple(output) != CONVNEXT_FAMILY_NAMES:
            raise RuntimeError("ConvNeXt feature family order differs from the release")
        for name, tokens, dimensions in CONVNEXT_FAMILY_SPECS:
            expected = (images.shape[0], tokens, dimensions)
            if tuple(output[name].shape) != expected:
                raise RuntimeError(
                    f"ConvNeXt family {name!r} shape {tuple(output[name].shape)} "
                    f"does not match {expected}"
                )
        return output


def _contract_file(root: Path, filename: str, size: int, digest: str) -> Path:
    path = root / filename
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"missing local ConvNeXt file: {path}")
    if path.stat().st_size != size:
        raise ValueError(f"local ConvNeXt file has wrong size: {filename}")
    if sha256_file(path) != digest:
        raise ValueError(f"local ConvNeXt file has wrong SHA-256: {filename}")
    return path


def _validate_hf_config(config_path: Path) -> None:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid local ConvNeXt config.json") from error
    expected = {
        "model_type": "convnextv2",
        "architectures": ["ConvNextV2ForImageClassification"],
        "depths": [3, 3, 27, 3],
        "hidden_sizes": [128, 256, 512, 1024],
        "image_size": 224,
        "patch_size": 4,
        "num_channels": 3,
    }
    differing = {
        key: {"expected": value, "found": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if differing:
        raise ValueError(f"local ConvNeXt config differs: {differing}")


def _signed_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    manifest.pop("manifest_sha256", None)
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _preflight_local_runtime() -> tuple[
    type[Any], type[Any], type[nn.Module], dict[str, Any]
]:
    """Fail closed on the complete imported runtime before ``from_pretrained``."""

    import transformers
    from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
    from transformers import activations as transformers_activations
    from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2GRN

    observed_versions = {
        "transformers": str(transformers.__version__),
        "torch": str(torch.__version__),
        "hip": None if torch.version.hip is None else str(torch.version.hip),
    }
    expected_versions = {
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
        "torch": EXPECTED_TORCH_VERSION,
        "hip": EXPECTED_HIP_VERSION,
    }
    if observed_versions != expected_versions:
        raise RuntimeError(
            "local ConvNeXt runtime versions differ from the attested adaptation: "
            f"expected {expected_versions}, found {observed_versions}"
        )

    source_paths = {
        "configuration_convnextv2.py": Path(
            inspect.getsourcefile(ConvNextV2Config) or ""
        ),
        "modeling_convnextv2.py": Path(
            inspect.getsourcefile(ConvNextV2ForImageClassification) or ""
        ),
        "activations.py": Path(transformers_activations.__file__ or ""),
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise RuntimeError("could not resolve every imported Transformers source file")
    observed_source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    if observed_source_hashes != EXPECTED_TRANSFORMERS_SOURCE_SHA256:
        raise RuntimeError(
            "imported Transformers sources differ from the attested adaptation: "
            f"expected {EXPECTED_TRANSFORMERS_SOURCE_SHA256}, "
            f"found {observed_source_hashes}"
        )

    wrapper_path = Path(__file__).resolve()
    if not wrapper_path.is_file():
        raise RuntimeError("could not attest the local ConvNeXt wrapper")
    manifest = _signed_manifest(
        {
            "adaptation_id": LOCAL_RUNTIME_ADAPTATION,
            "status": "local_pytorch_adaptation_of_official_drifting_jax",
            "not_a_native_transformers_5_3_reproduction": True,
            "versions": observed_versions,
            "transformers_sources": {
                name: {
                    "path": str(source_paths[name].resolve()),
                    "sha256": observed_source_hashes[name],
                }
                for name in sorted(source_paths)
            },
            "wrapper_source": {
                "path": str(wrapper_path),
                "sha256": sha256_file(wrapper_path),
            },
            "semantic_corrections": {
                "grn": "sqrt(sum(x**2)+1e-6)/(mean(sqrt(sum(x**2)+1e-6))+1e-6)",
                "expected_replacement_count": CONVNEXT_GRN_COUNT,
                "final_layer_norm_eps": 1e-6,
            },
        }
    )
    return (
        ConvNextV2Config,
        ConvNextV2ForImageClassification,
        ConvNextV2GRN,
        manifest,
    )


def attest_local_convnext_runtime() -> dict[str, Any]:
    """Return the signed, fail-closed runtime attestation without loading weights."""

    *_, manifest = _preflight_local_runtime()
    return manifest


def _strict_loading_report(loading_info: Any) -> dict[str, list[Any]]:
    required = {
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    }
    if not isinstance(loading_info, Mapping) or set(loading_info) != required:
        found = set(loading_info) if isinstance(loading_info, Mapping) else type(loading_info)
        raise RuntimeError(
            f"unexpected Transformers loading-report schema: expected {required}, found {found}"
        )
    nonempty = {key: value for key, value in loading_info.items() if value}
    if nonempty:
        raise RuntimeError(f"non-empty strict ConvNeXt loading report: {nonempty}")
    return {key: [] for key in sorted(required)}


def load_local_convnextv2_base(
    snapshot_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    model_id: str = HF_MODEL_ID,
    revision: str = HF_MODEL_REVISION,
    expected_config_sha256: str = HF_CONFIG_SHA256,
    expected_weights_sha256: str = HF_WEIGHTS_SHA256,
) -> CelebA64ConvNextV2Features:
    """Verify and load the pinned HF model from a local snapshot only.

    The identity arguments are intentionally closed rather than generic.  A
    different revision or weight is a new scientific intervention and must get
    a separately versioned loader/contract.  This function never calls
    ``snapshot_download`` and passes ``local_files_only=True`` to Transformers.
    """

    contract = {
        "model_id": (model_id, HF_MODEL_ID),
        "revision": (revision, HF_MODEL_REVISION),
        "config_sha256": (expected_config_sha256, HF_CONFIG_SHA256),
        "weights_sha256": (expected_weights_sha256, HF_WEIGHTS_SHA256),
    }
    differing = {
        name: {"expected": expected, "found": found}
        for name, (found, expected) in contract.items()
        if found != expected
    }
    if differing:
        raise ValueError(f"ConvNeXt model identity is not the pinned contract: {differing}")

    root = Path(snapshot_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"local ConvNeXt snapshot is absent: {root}")
    config_path = _contract_file(
        root, HF_CONFIG_FILENAME, HF_CONFIG_SIZE_BYTES, expected_config_sha256
    )
    weights_path = _contract_file(
        root, HF_WEIGHTS_FILENAME, HF_WEIGHTS_SIZE_BYTES, expected_weights_sha256
    )
    _validate_hf_config(config_path)

    # The delayed import keeps lightweight tests independent of Transformers.
    # Runtime versions and all three executed Transformers source files are
    # attested before either config or model calls ``from_pretrained``.
    (
        ConvNextV2Config,
        ConvNextV2ForImageClassification,
        ConvNextV2GRN,
        runtime_manifest,
    ) = _preflight_local_runtime()
    runtime_manifest = dict(runtime_manifest)
    runtime_manifest.pop("manifest_sha256")
    runtime_manifest["snapshot_files"] = {
        HF_CONFIG_FILENAME: {
            "path": str(config_path.resolve()),
            "size_bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        HF_WEIGHTS_FILENAME: {
            "path": str(weights_path.resolve()),
            "size_bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path),
        },
    }
    runtime_manifest = _signed_manifest(runtime_manifest)

    config = ConvNextV2Config.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    # HF's pinned config says 1e-12 for the final norm; the release's custom JAX
    # ConvNeXt uses 1e-6.  Other HF ConvNeXt-V2 norms are already hard-coded to
    # 1e-6, so this restores the one material epsilon mismatch.
    config.layer_norm_eps = 1e-6
    loaded = ConvNextV2ForImageClassification.from_pretrained(
        str(root),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Transformers did not return the requested loading report")
    classification_model, loading_info = loaded
    loading_report = _strict_loading_report(loading_info)
    backbone = classification_model.convnextv2
    grn_replacement_count = replace_hf_grn_with_official_jax(
        backbone,
        ConvNextV2GRN,
        expected_count=CONVNEXT_GRN_COUNT,
    )
    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    if parameter_count != CONVNEXT_BACKBONE_PARAMETER_COUNT:
        raise RuntimeError(
            f"ConvNeXt backbone has {parameter_count} parameters, expected "
            f"{CONVNEXT_BACKBONE_PARAMETER_COUNT}"
        )
    parameter_dtypes = {parameter.dtype for parameter in backbone.parameters()}
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(
            f"ConvNeXt backbone must load entirely in FP32, found {parameter_dtypes}"
        )
    if float(backbone.layernorm.eps) != 1e-6:
        raise RuntimeError("ConvNeXt final LayerNorm epsilon is not release-compatible")

    identity = {
        "model_id": model_id,
        "revision": revision,
        "snapshot_dir": str(root.resolve()),
        "config": {
            "filename": HF_CONFIG_FILENAME,
            "size_bytes": HF_CONFIG_SIZE_BYTES,
            "sha256": expected_config_sha256,
        },
        "weights": {
            "filename": HF_WEIGHTS_FILENAME,
            "size_bytes": HF_WEIGHTS_SIZE_BYTES,
            "sha256": expected_weights_sha256,
        },
        "runtime_manifest": runtime_manifest,
        "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
        "strict_loading_report": loading_report,
        "backbone_parameter_count": parameter_count,
        "compute_dtype": "float32",
        "final_layer_norm_eps": 1e-6,
        "official_jax_grn_eps": OFFICIAL_GRN_EPS,
        "official_jax_grn_replacement_count": grn_replacement_count,
        "official_release_commit": OFFICIAL_RELEASE_COMMIT,
        "feature_family_names": list(CONVNEXT_FAMILY_NAMES),
    }
    adapter = CelebA64ConvNextV2Features(backbone, model_identity=identity)
    adapter.to(device=device)
    adapter.eval()
    return adapter


def _validate_chunk_size(chunk_size: int, label: str) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"{label} must be a positive integer")


def _extract_chunked(
    feature_encoder: CelebA64ConvNextV2Features,
    images: Tensor,
    *,
    chunk_size: int,
    checkpoint_chunks: bool,
) -> dict[str, Tensor]:
    feature_encoder._assert_frozen_eval()
    _validate_images(images)
    _validate_chunk_size(chunk_size, "chunk_size")
    if not isinstance(checkpoint_chunks, bool):
        raise TypeError("checkpoint_chunks must be boolean")

    collected: dict[str, list[Tensor]] = {
        name: [] for name in CONVNEXT_FAMILY_NAMES
    }

    def tuple_forward(chunk: Tensor) -> tuple[Tensor, ...]:
        features = feature_encoder(chunk)
        return tuple(features[name] for name in CONVNEXT_FAMILY_NAMES)

    for start in range(0, images.shape[0], chunk_size):
        chunk = images[start : start + chunk_size]
        if checkpoint_chunks and torch.is_grad_enabled() and chunk.requires_grad:
            values = activation_checkpoint(
                tuple_forward,
                chunk,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            values = tuple_forward(chunk)
        for name, value in zip(CONVNEXT_FAMILY_NAMES, values):
            collected[name].append(value)

    return {
        name: torch.cat(collected[name], dim=0) for name in CONVNEXT_FAMILY_NAMES
    }


def extract_generated_convnext_features(
    feature_encoder: CelebA64ConvNextV2Features,
    generated_images: Tensor,
    *,
    chunk_size: int = 8,
    checkpoint_chunks: bool = True,
) -> dict[str, Tensor]:
    """Extract chunks without detaching the generated-image Jacobian.

    The caller must perform the existing differentiable cross-rank all-gather
    *before* this function.  Chunking only feature extraction is exact; the GMD
    loss must still be evaluated once on the concatenated global support.
    """

    return _extract_chunked(
        feature_encoder,
        generated_images,
        chunk_size=chunk_size,
        checkpoint_chunks=checkpoint_chunks,
    )


def extract_fixed_convnext_features(
    feature_encoder: CelebA64ConvNextV2Features,
    fixed_images: Tensor,
    *,
    chunk_size: int = 32,
) -> dict[str, Tensor]:
    """Extract real positive/negative supports without an autograd graph."""

    # Do not use inference_mode: its tensors can be illegal when an autograd
    # GMD operation later needs to save them for the generated-image backward.
    with torch.no_grad():
        return _extract_chunked(
            feature_encoder,
            fixed_images.detach(),
            chunk_size=chunk_size,
            checkpoint_chunks=False,
        )


def extract_convnext_supports(
    feature_encoder: CelebA64ConvNextV2Features,
    global_generated_images: Tensor,
    positive_images: Tensor,
    fixed_negative_images: Tensor,
    *,
    generated_chunk_size: int = 8,
    fixed_chunk_size: int = 32,
    checkpoint_generated: bool = True,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Return generated, positive, and negative dictionaries for one GMD call."""

    generated = extract_generated_convnext_features(
        feature_encoder,
        global_generated_images,
        chunk_size=generated_chunk_size,
        checkpoint_chunks=checkpoint_generated,
    )
    positive = extract_fixed_convnext_features(
        feature_encoder, positive_images, chunk_size=fixed_chunk_size
    )
    negative = extract_fixed_convnext_features(
        feature_encoder, fixed_negative_images, chunk_size=fixed_chunk_size
    )
    if not (tuple(generated) == tuple(positive) == tuple(negative)):
        raise RuntimeError("ConvNeXt support feature families differ")
    return generated, positive, negative


def merge_feature_families(
    existing: Mapping[str, Tensor],
    convnext: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Append the 13 ConvNeXt families without overwriting MAE families."""

    if tuple(convnext) != CONVNEXT_FAMILY_NAMES:
        raise ValueError("ConvNeXt feature names/order do not match the contract")
    collisions = sorted(set(existing).intersection(convnext))
    if collisions:
        raise ValueError(f"feature-family collision: {collisions}")
    merged = dict(existing)
    merged.update(convnext)
    return merged


__all__ = [
    "CELEBA_RGB_SHAPE",
    "CONVNEXT_BACKBONE_PARAMETER_COUNT",
    "CONVNEXT_FAMILY_NAMES",
    "CONVNEXT_FAMILY_SPECS",
    "CONVNEXT_GRN_COUNT",
    "EXPECTED_HIP_VERSION",
    "EXPECTED_TORCH_VERSION",
    "EXPECTED_TRANSFORMERS_SOURCE_SHA256",
    "EXPECTED_TRANSFORMERS_VERSION",
    "HF_CONFIG_FILENAME",
    "HF_CONFIG_SHA256",
    "HF_CONFIG_SIZE_BYTES",
    "HF_MODEL_ID",
    "HF_MODEL_REVISION",
    "HF_WEIGHTS_FILENAME",
    "HF_WEIGHTS_SHA256",
    "HF_WEIGHTS_SIZE_BYTES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LOCAL_RUNTIME_ADAPTATION",
    "OFFICIAL_RELEASE_COMMIT",
    "OFFICIAL_GRN_EPS",
    "CelebA64ConvNextV2Features",
    "OfficialJaxConvNextV2GRN",
    "attest_local_convnext_runtime",
    "extract_convnext_supports",
    "extract_fixed_convnext_features",
    "extract_generated_convnext_features",
    "load_local_convnextv2_base",
    "merge_feature_families",
    "normalize_convnext_stage",
    "preprocess_rgb64_for_convnext",
    "replace_hf_grn_with_official_jax",
    "safe_std",
    "sha256_file",
]
