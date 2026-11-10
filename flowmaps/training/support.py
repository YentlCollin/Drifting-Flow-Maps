"""Shared optimization, data-stream and CLI helpers for the RGB trainer."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from flowmaps.data.images import CelebA64PixelDataset
from flowmaps.features.mae import canonical_json_sha256
from flowmaps.models.backbone import (
    DEFAULT_CONDITION_TOKENS,
    DEFAULT_DEPTH,
    DEFAULT_HEADS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MLP_RATIO,
    DEFAULT_NOISE_CLASSES,
    DEFAULT_NOISE_COORDS,
    DEFAULT_PATCH_SIZE,
    CelebA64DirectPixelGMDDiT,
)
from flowmaps.paths import REPO_ROOT

AUGMENTATIONS = ("aligned_flip", "none")


DEFAULT_TEMPERATURES = (0.2, 0.05, 0.02)


PRODUCTION_FEATURE_BASE_CHANNELS = 64


PRODUCTION_FEATURE_LAYERS = [3, 4, 6, 3]


PRODUCTION_FEATURE_PROFILE = "nondegenerate"


PRODUCTION_FEATURE_MINIMUM_UPDATES = 10_000


PRODUCTION_FEATURE_ARCHITECTURE = {
    "base_channels": PRODUCTION_FEATURE_BASE_CHANNELS,
    "layers": PRODUCTION_FEATURE_LAYERS,
    "dropout_prob": 0.0,
    "gn_max_groups": 32,
    "input_patch_size": 2,
    "mask_patch_size": 2,
    "feature_patch_sizes": [2, 4],
    "every_k_block": 2,
    "feature_profile": PRODUCTION_FEATURE_PROFILE,
}


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_temperatures(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(value.strip()) for value in raw.split(",") if value.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "temperatures must be comma-separated floats"
        ) from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("temperatures must be finite and positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct RGB CelebA64 generator trained with released GMD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "celeba64" / "celeba64_train",
    )
    parser.add_argument("--train-start-index", type=int, default=5_000)
    parser.add_argument("--train-end-index", type=int, default=30_000)
    parser.add_argument("--feature-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-checkpoint-sha256", required=True)
    parser.add_argument("--feature-weight-kind", choices=("ema", "raw"), default="ema")
    parser.add_argument(
        "--expected-feature-training-signature-sha256",
        default=None,
        help=(
            "required outside --smoke; pins the exact audited phase-2 training "
            "recipe independently of the artifact hash"
        ),
    )
    parser.add_argument(
        "--allow-short-feature-checkpoint-preflight",
        action="store_true",
        help=(
            "GPU preflight only: relax the 10k-update feature floor while retaining "
            "the exact production architecture and EMA requirement"
        ),
    )
    parser.add_argument(
        "--expected-source-manifest",
        type=Path,
        default=None,
        help="launcher-created pre-import SHA-256 manifest for all imported training sources",
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=None,
        help="external SHA-256 pin for --expected-source-manifest",
    )

    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument(
        "--global-generated",
        type=int,
        default=64,
        help="complete generated support across all ranks",
    )
    parser.add_argument("--positive-count", type=int, default=128)
    parser.add_argument("--positive-bank-size", type=int, default=128)
    parser.add_argument("--negative-count", type=int, default=32)
    parser.add_argument("--negative-bank-size", type=int, default=1_000)
    parser.add_argument("--bank-push", type=int, default=128)
    parser.add_argument("--augmentation", choices=AUGMENTATIONS, default="aligned_flip")
    parser.add_argument(
        "--temperatures", type=_parse_temperatures, default=DEFAULT_TEMPERATURES
    )

    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--mlp-ratio", type=float, default=DEFAULT_MLP_RATIO)
    parser.add_argument("--noise-classes", type=int, default=DEFAULT_NOISE_CLASSES)
    parser.add_argument("--noise-coords", type=int, default=DEFAULT_NOISE_COORDS)
    parser.add_argument(
        "--condition-tokens", type=int, default=DEFAULT_CONDITION_TOKENS
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-warmup-start", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=10_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--use-bfloat16", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--preload-uint8", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--distributed-backend", choices=("nccl", "gloo"), default="nccl"
    )
    parser.add_argument("--distributed-timeout-minutes", type=int, default=30)

    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "celeba_pixel_gmd" / "generator",
    )
    parser.add_argument("--run-label", default="celeba64_direct_pixel_gmd")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-sha256", default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument(
        "--verify-gradients",
        action="store_true",
        help=(
            "run the detailed complete/finite/DDP gradient fingerprint at global "
            "step 1 and then at --gradient-verification-interval"
        ),
    )
    parser.add_argument(
        "--gradient-verification-interval",
        type=int,
        default=0,
        help="0 disables periodic detailed checks after global step 1",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU/Gloo configuration; still executes the real frozen-MAE GMD path",
    )
    return parser


def seed_parameters(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def make_prior_generator(args: argparse.Namespace) -> torch.Generator:
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed + 1_000_003 * args.rank + 17)
    return generator


def make_bank_generator(args: argparse.Namespace) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(args.seed + 29)


class GlobalBankStream:
    """Identical deterministic dataset permutation on every rank."""

    def __init__(
        self, dataset: CelebA64PixelDataset, *, batch_size: int, seed: int
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.batches_per_epoch = len(dataset) // self.batch_size
        if self.batches_per_epoch < 1:
            raise ValueError("training split is smaller than one bank-push batch")
        self.usable_count = self.batches_per_epoch * self.batch_size
        self.epoch = 0
        self.batch_offset = 0
        self._permutation = self._make_permutation()

    def _make_permutation(self) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator)[
            : self.usable_count
        ]

    def next_indices(self) -> Tensor:
        if self.batch_offset == self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0
            self._permutation = self._make_permutation()
        start = self.batch_offset * self.batch_size
        indices = self._permutation[start : start + self.batch_size]
        self.batch_offset += 1
        return indices

    def next(self) -> Tensor:
        return self.dataset.batch(self.next_indices())

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"epoch", "batch_offset"}:
            raise ValueError("malformed bank stream state")
        epoch, offset = int(state["epoch"]), int(state["batch_offset"])
        if epoch < 0 or not 0 <= offset <= self.batches_per_epoch:
            raise ValueError("invalid bank stream position")
        self.epoch, self.batch_offset = epoch, offset
        self._permutation = self._make_permutation()


def augment_bank_images(
    images_uint8: Tensor,
    mode: str,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Map aligned uint8 files to FP32 [-1,1] with an explicit CPU RNG."""

    if images_uint8.device.type != "cpu" or images_uint8.dtype != torch.uint8:
        raise ValueError("bank augmentation expects CPU uint8 images")
    if mode not in AUGMENTATIONS:
        raise ValueError("unsupported bank augmentation")
    images = images_uint8.float().div_(255.0)
    if mode == "aligned_flip":
        flip = torch.rand(images.shape[0], generator=generator) < 0.5
        images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    return images.mul_(2.0).sub_(1.0)


def learning_rate_at_step(step_index: int, args: argparse.Namespace) -> float:
    if args.lr_warmup_steps and step_index < args.lr_warmup_steps:
        progress = step_index / args.lr_warmup_steps
        return args.lr_warmup_start + progress * (args.lr - args.lr_warmup_start)
    return args.lr


@torch.no_grad()
def update_ema(
    ema: CelebA64DirectPixelGMDDiT,
    model: CelebA64DirectPixelGMDDiT,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(
        ema.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def feature_checkpoint_identity(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(args.feature_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("feature checkpoint payload is not a mapping")
    architecture = payload.get("architecture")
    schema = payload.get("feature_schema")
    metadata = payload.get("metadata")
    integrity = payload.get("integrity")
    if not all(
        isinstance(value, Mapping)
        for value in (architecture, schema, metadata, integrity)
    ):
        raise ValueError(
            "feature checkpoint lacks strict architecture/schema/metadata/integrity"
        )
    required_metadata = {
        "completed_step",
        "training_signature",
        "training_signature_sha256",
        "weight_kind",
    }
    if not required_metadata.issubset(metadata):
        raise ValueError(
            "feature checkpoint is not an exported phase-2 training artifact"
        )
    completed_step = metadata["completed_step"]
    feature_training_signature = metadata["training_signature"]
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise ValueError("feature checkpoint completed_step is malformed")
    if completed_step < 1:
        raise ValueError("feature checkpoint has no completed phase-2 update")
    if not isinstance(feature_training_signature, Mapping):
        raise ValueError("feature checkpoint phase-2 signature is malformed")
    if (
        feature_training_signature.get("objective")
        != "masked_only_packed_pixel_reconstruction"
    ):
        raise ValueError(
            "feature checkpoint was not trained with the phase-2 MAE objective"
        )
    if metadata["training_signature_sha256"] != canonical_json_sha256(
        dict(feature_training_signature)
    ):
        raise ValueError(
            "feature checkpoint phase-2 training signature SHA-256 mismatch"
        )
    expected_training_signature = args.expected_feature_training_signature_sha256
    if not args.smoke and not _valid_sha256(expected_training_signature):
        raise ValueError(
            "non-smoke feature identity requires the expected phase-2 training signature SHA-256"
        )
    if (
        expected_training_signature is not None
        and metadata["training_signature_sha256"] != expected_training_signature
    ):
        raise ValueError(
            "feature checkpoint phase-2 training signature differs from the pinned expectation"
        )
    if metadata["weight_kind"] != args.feature_weight_kind:
        raise ValueError(
            f"feature checkpoint weight_kind={metadata['weight_kind']!r}, "
            f"expected {args.feature_weight_kind!r}"
        )
    if metadata["weight_kind"] != "ema":
        raise ValueError("the production pixel-GMD contract requires EMA features")
    production_architecture = PRODUCTION_FEATURE_ARCHITECTURE
    architecture_mismatches = {
        key: {"expected": expected, "observed": architecture.get(key)}
        for key, expected in production_architecture.items()
        if architecture.get(key) != expected
    }
    if architecture_mismatches and not args.smoke:
        raise ValueError(
            "feature checkpoint violates the production MAE architecture contract: "
            f"{architecture_mismatches}"
        )
    short_checkpoint = completed_step < PRODUCTION_FEATURE_MINIMUM_UPDATES
    if short_checkpoint and not (
        args.smoke or args.allow_short_feature_checkpoint_preflight
    ):
        raise ValueError(
            "feature checkpoint is shorter than the production 10000-update floor; "
            "only an explicit <=2-step GPU preflight may relax it"
        )
    return {
        "file_sha256": args.feature_checkpoint_sha256,
        "weight_kind": metadata["weight_kind"],
        "completed_step": completed_step,
        "phase2_training_signature_sha256": metadata["training_signature_sha256"],
        "expected_phase2_training_signature_sha256": expected_training_signature,
        "architecture": dict(architecture),
        "feature_schema": dict(schema),
        "state_dict_sha256": integrity.get("state_dict_sha256"),
        "production_contract": {
            "weight_kind": "ema",
            **production_architecture,
            "minimum_completed_step": PRODUCTION_FEATURE_MINIMUM_UPDATES,
            "architecture_satisfied": not architecture_mismatches,
            "updates_satisfied": not short_checkpoint,
            "short_checkpoint_preflight_opt_in": bool(
                args.allow_short_feature_checkpoint_preflight
            ),
            "unit_smoke_exception": bool(args.smoke),
        },
    }


def reduce_stats(
    stats: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Tensor]:
    keys = list(stats)
    packed = torch.stack(
        [
            value.detach().to(args.device, dtype=torch.float32).reshape(())
            if torch.is_tensor(value)
            else torch.tensor(float(value), device=args.device)
            for value in stats.values()
        ]
    )
    if args.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(args.world_size)
    return {name: packed[index] for index, name in enumerate(keys)}


def reduce_max_stats(
    stats: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Tensor]:
    """Return scalar maxima across ranks for timing and memory telemetry."""

    keys = list(stats)
    packed = torch.stack(
        [
            value.detach().to(args.device, dtype=torch.float64).reshape(())
            if torch.is_tensor(value)
            else torch.tensor(float(value), device=args.device, dtype=torch.float64)
            for value in stats.values()
        ]
    )
    if args.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return {name: packed[index] for index, name in enumerate(keys)}


def compact_gmd_statistics(statistics: Mapping[str, Tensor]) -> dict[str, Tensor]:
    scales = [value for key, value in statistics.items() if key.startswith("scale/")]
    energies = [
        value for key, value in statistics.items() if key.startswith("force_energy_")
    ]
    family_losses = [
        value for key, value in statistics.items() if key.startswith("loss/")
    ]
    if not scales or not energies or not family_losses:
        raise RuntimeError("GMD feature statistics are incomplete")
    return {
        "gmd_loss": statistics["loss"],
        "feature_family_count": float(len(family_losses)),
        "family_loss_mean": torch.stack(family_losses).mean(),
        "family_loss_max": torch.stack(family_losses).max(),
        "feature_scale_mean": torch.stack(scales).mean(),
        "force_energy_mean": torch.stack(energies).mean(),
        "generated_support": statistics["support/generated_global"],
        "positive_support": statistics["support/positive"],
        "fixed_negative_support": statistics["support/fixed_negative"],
        "fixed_negative_weight_sum": statistics["support/fixed_negative_weight_sum"],
    }
