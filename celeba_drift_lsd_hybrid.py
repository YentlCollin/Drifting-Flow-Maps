#!/usr/bin/env python3
"""Train CelebA variant D: mixed LSD plus a terminal official-GMD anchor.

This driver intentionally imports the attested pure-LSD machinery instead of
forking its data stream, optimizer, EMA, checkpoint, or exact 75/25 loss.  A
long run must receive an explicit ``--gmd-lambda`` fixed from a prior
calibration smoke.  Only a run explicitly labelled ``--calibration-smoke``
may derive and immediately exercise the equal-RMS-gradient value.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import shutil
import sys
from collections.abc import Mapping
from typing import Any

# Import the baseline runner before torch.  Its module-level bootstrap creates
# one MIOpen cache directory per local rank before PyTorch/ROCm is initialized;
# importing torch first would silently defeat that multi-rank safety contract.
import celeba_latent_lsd as lsd_runner
import torch
from torch import Tensor, nn

from local_paths import REPO_ROOT
from src.celeba_drift_lsd_hybrid import (
    DriftLSDHybridTrainingModel,
    distributed_component_gradient_moments,
    equal_rms_gradient_calibration,
)
from src.celeba_gmd_features_v2 import (
    CHECKPOINT_FORMAT as FEATURE_CHECKPOINT_FORMAT,
    FEATURE_PROFILES,
    FEATURE_PROFILE_OFFICIAL_70,
    load_frozen_feature_encoder_v2,
)


HYBRID_SOURCE_FILES = (
    'celeba_drift_lsd_hybrid.py',
    'src/celeba_drift_lsd_hybrid.py',
    'src/celeba_gmd_features_v2.py',
    'src/official_drifting.py',
)


# The scientific variant-D comparison changes only the objective relative to
# the pre-specified variant-A screen.  Keep that claim executable: production
# runs fail closed if any matched setting drifts.  CPU ``--smoke`` runs and
# GPU ``--calibration-smoke`` pilots both record that they are non-scientific
# comparisons.
BASELINE_A_EXPECTED_SETTINGS: dict[str, Any] = {
    "steps": 30_000,
    "train_start_index": 5_000,
    "train_end_index": 30_000,
    "global_batch": 256,
    "world_size": 8,
    "local_batch": 32,
    "source_std": 1.0,
    "coupling": "independent",
    "gmd_teacher_checkpoint": None,
    "lsd_target": "current",
    "base_channels": 256,
    "time_embedding_dim": 32,
    "condition_dim": None,
    "sigma_data": 0.5,
    "zero_init_velocity": True,
    "logvar_channels": 128,
    "use_bfloat16": True,
    "lr": 1e-2,
    "lr_decay_steps": 35_000,
    "radam_beta1": 0.9,
    "radam_beta2": 0.999,
    "radam_eps": 1e-8,
    "grad_clip": 1.0,
    "ema_decay": 0.9999,
    "seed": 42,
    "deterministic": True,
    # A scientific comparison must train normally for the full pre-specified
    # budget.  Calibration is a separate pilot and can never inherit the
    # "only objective addition" claim.
    "gradient_calibration_batches": 0,
    "calibration_smoke": False,
}
BASELINE_A_EXPECTED_DATA: dict[str, str] = {
    "cache_sha256": "a2ebb16a250444652ed604b8e693cd21b09a9214df4ba89a2290b674dc3c0ca2",
    "train_values_sha256": "7609dd474070c4e04103ace1b00ff44a67726242d651c3ba2a64c3e4f491fa89",
}


def baseline_a_comparison_contract(args: argparse.Namespace) -> dict[str, Any]:
    settings = {
        name: {
            "actual": getattr(args, name),
            "expected": expected,
            "matched": getattr(args, name) == expected,
        }
        for name, expected in BASELINE_A_EXPECTED_SETTINGS.items()
    }
    settings.update(
        {
            name: {
                "actual": getattr(args, f"_baseline_a_{name}", None),
                "expected": expected,
                "matched": getattr(args, f"_baseline_a_{name}", None) == expected,
            }
            for name, expected in BASELINE_A_EXPECTED_DATA.items()
        }
    )
    all_matched = all(item["matched"] for item in settings.values())
    strict = not args.smoke and not args.calibration_smoke
    return {
        "baseline": "A_lsd_independent_std1_base256_seed42_screen",
        "settings": settings,
        "strict_match_enforced": strict,
        "all_matched": all_matched,
        "only_objective_addition": all_matched,
        "only_objective_addition_claim": (
            "terminal official MAE-v2 GMD anchor" if all_matched else None
        ),
    }


def parse_temperatures(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(piece.strip()) for piece in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GMD temperatures must be comma-separated floats") from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("GMD temperatures must be finite and positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GMD temperatures must not contain duplicates")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = lsd_runner.build_parser()
    parser.description = "CelebA latent drift-LSD hybrid (variant D)"
    parser.set_defaults(
        output_root=REPO_ROOT / "experiments" / "celeba_drift_lsd_hybrid",
        run_label="drift_lsd_hybrid",
    )
    parser.add_argument("--feature-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=FEATURE_PROFILE_OFFICIAL_70,
    )
    parser.add_argument(
        "--gmd-temperatures",
        type=parse_temperatures,
        default=parse_temperatures("0.2,0.05,0.02"),
    )
    parser.add_argument(
        "--gmd-lambda",
        type=float,
        default=None,
        help="fixed terminal-GMD weight; mandatory outside a calibration smoke",
    )
    parser.add_argument(
        "--gradient-calibration-batches",
        type=int,
        default=0,
        help="number of untouched initial batches used for component-gradient calibration",
    )
    parser.add_argument(
        "--calibration-smoke",
        action="store_true",
        help="allow this short run to derive lambda from equal DDP-averaged RMS gradients",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    invoked = sys.argv[1:] if argv is None else argv
    args.command = shlex.join([sys.argv[0], *invoked])
    # Keep the pure-LSD CPU smoke transformation byte-for-byte equivalent in
    # semantics; the production ROCm launcher uses --calibration-smoke only.
    if args.smoke:
        args.device = "cpu"
        args.distributed_backend = "gloo"
        args.steps = min(args.steps, 2)
        args.base_channels = min(args.base_channels, 8)
        args.time_embedding_dim = min(args.time_embedding_dim, 8)
        args.condition_dim = 16 if args.condition_dim is None else min(args.condition_dim, 16)
        args.logvar_channels = min(args.logvar_channels, 8)
        args.use_bfloat16 = False
        args.pin_memory = False
        args.train_start_index = 0
        args.train_end_index = None
        args.log_every = 1
        args.checkpoint_every = 1
    return args


def validate_hybrid_args(args: argparse.Namespace) -> None:
    lsd_runner.validate_args(args)
    if not args.feature_checkpoint.is_file():
        raise FileNotFoundError(f"v2 MAE checkpoint not found: {args.feature_checkpoint}")
    if args.source_std != 1.0:
        raise ValueError("variant D must keep A's standard-normal source (--source-std 1)")
    if args.coupling != "independent" or args.gmd_teacher_checkpoint is not None:
        raise ValueError("variant D keeps A's independent data coupling and has no endpoint teacher")
    if args.lsd_target != "current":
        raise ValueError("variant D keeps the released current-parameter stop-gradient target")
    if args.gradient_calibration_batches < 0:
        raise ValueError("--gradient-calibration-batches must be non-negative")
    if args.gmd_lambda is not None and (
        not math.isfinite(args.gmd_lambda) or args.gmd_lambda <= 0
    ):
        raise ValueError("--gmd-lambda must be finite and positive")
    if args.resume is not None and args.gradient_calibration_batches:
        raise ValueError("resume must reuse a fixed lambda and cannot rerun calibration")
    if args.calibration_smoke:
        if args.resume is not None:
            raise ValueError("a calibration smoke cannot resume")
        if args.gradient_calibration_batches < 1:
            raise ValueError("a calibration smoke requires at least one calibration batch")
        if args.gradient_calibration_batches > 8:
            raise ValueError("a calibration smoke is restricted to at most eight batches")
        if args.steps > 50:
            raise ValueError("--calibration-smoke is restricted to at most 50 updates")
    elif args.gmd_lambda is None:
        raise ValueError(
            "long/non-calibration runs require an explicit --gmd-lambda from a completed smoke"
        )
    if args.gmd_lambda is None and not args.calibration_smoke:
        raise ValueError("only --calibration-smoke may derive --gmd-lambda")
    comparison = baseline_a_comparison_contract(args)
    argument_mismatches = [
        name
        for name in BASELINE_A_EXPECTED_SETTINGS
        if not comparison["settings"][name]["matched"]
    ]
    if comparison["strict_match_enforced"] and argument_mismatches:
        raise ValueError(
            "scientific variant D must match pre-specified baseline A settings; "
            f"mismatches: {argument_mismatches}"
        )


def validate_baseline_a_data_identity(
    args: argparse.Namespace,
    *,
    cache_sha256: str,
    train_values_sha256: str,
) -> None:
    args._baseline_a_cache_sha256 = cache_sha256
    args._baseline_a_train_values_sha256 = train_values_sha256
    comparison = baseline_a_comparison_contract(args)
    if comparison["strict_match_enforced"] and not comparison["all_matched"]:
        mismatches = [
            name
            for name, setting in comparison["settings"].items()
            if not setting["matched"]
        ]
        raise ValueError(
            "scientific variant D must match pre-specified baseline A settings; "
            f"mismatches: {mismatches}"
        )


def _expected_feature_metadata(
    args: argparse.Namespace,
    cache_metadata: Mapping[str, Any],
    cache_sha256: str,
) -> dict[str, Any]:
    return {
        "cache_sha256": cache_sha256,
        "cache_model_id": cache_metadata.get("model_id"),
        "cache_encoding": cache_metadata.get("encoding"),
        "cache_scaling_factor": cache_metadata.get("scaling_factor"),
        "train_start_index": args.train_start_index,
        "train_end_index": args.train_end_index,
        "latent_normalization": "none",
    }


def load_feature_encoder(
    args: argparse.Namespace,
    cache_metadata: Mapping[str, Any],
    cache_sha256: str,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(args.feature_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != FEATURE_CHECKPOINT_FORMAT:
        raise ValueError("variant D requires a versioned CelebA MAE-v2 checkpoint")
    architecture = payload.get("architecture")
    metadata = payload.get("metadata")
    if not isinstance(architecture, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("MAE-v2 checkpoint lacks architecture or training metadata")
    if architecture.get("feature_profile") != args.feature_profile:
        raise ValueError(
            "feature checkpoint profile disagrees with --feature-profile: "
            f"{architecture.get('feature_profile')!r} != {args.feature_profile!r}"
        )
    expected = _expected_feature_metadata(args, cache_metadata, cache_sha256)
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f"MAE-v2/data-coordinate provenance mismatch for {mismatches}")
    encoder = load_frozen_feature_encoder_v2(payload, args.device)
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("loaded MAE-v2 encoder is not frozen in eval mode")
    feature_metadata = {
        "path": str(args.feature_checkpoint.resolve()),
        "sha256": lsd_runner.sha256_file(args.feature_checkpoint),
        "checkpoint_format": FEATURE_CHECKPOINT_FORMAT,
        "architecture": dict(architecture),
        "training_metadata": dict(metadata),
        "frozen": True,
        "input_gradient_preserved": True,
    }
    return encoder, feature_metadata


def make_hybrid_model(
    args: argparse.Namespace,
    feature_encoder: nn.Module,
    *,
    gmd_lambda: float,
) -> DriftLSDHybridTrainingModel:
    pure_model = lsd_runner.make_training_model(args)
    return DriftLSDHybridTrainingModel(
        pure_model.flow_map,
        feature_encoder,
        gmd_lambda=gmd_lambda,
        temperatures=args.gmd_temperatures,
        distributed_statistics=args.distributed,
    )


def calibrate_lambda(
    model: DriftLSDHybridTrainingModel,
    training_latents: Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Measure without changing parameters, data position, or training RNG."""

    calibration_stream = lsd_runner.LatentBatchStream(training_latents, args)
    saved_rng = lsd_runner.gather_rng_states(args)
    moments: list[Mapping[str, float | int]] = []
    named_parameters = tuple(model.named_parameters())
    try:
        for batch_index in range(args.gradient_calibration_batches):
            x1 = calibration_stream.next().to(
                args.device, non_blocking=args.device.type == "cuda"
            )
            x0 = args.source_std * torch.randn_like(x1)
            with torch.autocast(
                device_type=args.device.type,
                dtype=torch.bfloat16,
                enabled=args.use_bfloat16 and args.device.type == "cuda",
            ):
                terms = model.objective_terms(x0, x1, model.flow_map)
            if terms.statistics["diagonal_count"] != 3 * args.local_batch // 4:
                raise RuntimeError("calibration did not preserve the exact local 75/25 split")
            if terms.statistics["offdiagonal_count"] != args.local_batch // 4:
                raise RuntimeError("calibration did not preserve the exact local 75/25 split")
            batch = distributed_component_gradient_moments(
                terms.lsd,
                terms.gmd,
                named_parameters,
                distributed=args.distributed,
            )
            trainable = int(batch["trainable_parameter_tensors"])
            expected_shared = trainable - 1
            if int(batch["lsd_active_parameter_tensors"]) != trainable:
                raise RuntimeError(
                    "LSD calibration does not reach every trainable parameter tensor"
                )
            if (
                int(batch["gmd_active_parameter_tensors"]) != expected_shared
                or int(batch["shared_parameter_tensors"]) != expected_shared
            ):
                raise RuntimeError(
                    "terminal GMD must reach every trainable tensor except the single "
                    "LSD-only log-variance weight"
                )
            finite_fields = (
                "lsd_gradient_squared_norm",
                "gmd_gradient_squared_norm",
                "gradient_dot_product",
                "lsd_gradient_norm",
                "gmd_gradient_norm",
                "gradient_cosine",
                "shared_lsd_gradient_squared_norm",
                "shared_gmd_gradient_squared_norm",
                "shared_gradient_dot_product",
                "shared_lsd_gradient_norm",
                "shared_gmd_gradient_norm",
                "shared_gradient_cosine",
            )
            if any(
                batch[name] is None or not math.isfinite(float(batch[name]))
                for name in finite_fields
            ):
                raise RuntimeError("component-gradient calibration statistics are non-finite")
            moments.append({"batch_index": batch_index, **batch})
            del terms
    finally:
        lsd_runner.restore_rng_state(saved_rng, args)
    result = equal_rms_gradient_calibration(moments)
    if not math.isfinite(float(result["lambda_equal_rms"])) or float(
        result["lambda_equal_rms"]
    ) <= 0:
        raise RuntimeError("equal-RMS calibration produced an invalid GMD lambda")
    result.update(
        {
            "data_stream_consumed_by_training": False,
            "training_rng_restored": True,
            "model_parameters_updated": False,
            "component_loss_values_not_used_for_scaling": True,
        }
    )
    return result


def hybrid_source_hashes() -> dict[str, str]:
    missing = [name for name in HYBRID_SOURCE_FILES if not (REPO_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"hybrid training sources are missing: {missing}")
    return {
        name: lsd_runner.sha256_file(REPO_ROOT / name) for name in HYBRID_SOURCE_FILES
    }


def augment_hybrid_training_signature(
    signature: Mapping[str, Any],
    args: argparse.Namespace,
    feature_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    signature = dict(signature)
    comparison = baseline_a_comparison_contract(args)
    base_signature_version = signature.get("signature_version")
    signature.update(
        {
            "signature_version": 2,
            "base_lsd_signature_version": base_signature_version,
            "objective": "mixed_lsd_plus_terminal_official_gmd_mae_v2",
            "base_lsd_objective": "exact_75_25_diagonal_fm_and_lagrangian_self_distillation",
            "terminal_gmd_map": "X_theta(x0,0,1)",
            "terminal_gmd_lambda": float(args.gmd_lambda),
            "terminal_gmd_temperatures": list(args.gmd_temperatures),
            "terminal_gmd_feature_checkpoint_sha256": feature_metadata["sha256"],
            "terminal_gmd_feature_profile": args.feature_profile,
            "terminal_gmd_feature_dtype": "float32",
            "terminal_gmd_formula": "src.official_drifting.summed_feature_drifting_loss",
            "terminal_gmd_positive_stop_gradient": True,
            "terminal_gmd_generated_feature_input_gradient": True,
            "terminal_gmd_particle_support": "rank_local",
            "terminal_gmd_scalar_statistics": (
                "global_ddp_average" if args.distributed else "local"
            ),
            "ddp_boundary": (
                "one outer forward containing mixed LSD JVP/current-stopgrad/logvar "
                "and terminal MAE-v2 GMD"
            ),
            "hybrid_source_files_sha256": hybrid_source_hashes(),
            "baseline_a_all_matched": comparison["all_matched"],
            "baseline_a_strict_match_enforced": comparison["strict_match_enforced"],
            "baseline_a_comparison_contract": comparison,
        }
    )
    return signature


def archive_hybrid_sources(run_dir: Path, suffix: str) -> dict[str, Any]:
    destination = run_dir / f"hybrid_source_snapshot{suffix}"
    destination.mkdir(parents=True, exist_ok=False)
    hashes = hybrid_source_hashes()
    for name in HYBRID_SOURCE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
    manifest = {
        "created_at_utc": lsd_runner.utc_now(),
        "source_files_sha256": hashes,
    }
    lsd_runner.atomic_json_dump(destination / "manifest.json", manifest)
    return {"path": str(destination), **manifest}


def write_hybrid_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    feature_metadata: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
) -> None:
    suffix = "" if args.resume is None else f"_resume_{lsd_runner.utc_now().replace(':', '-')}"
    snapshot = archive_hybrid_sources(run_dir, suffix)
    comparison = baseline_a_comparison_contract(args)
    payload = {
        "format": "driftmodel.celeba_drift_lsd_hybrid_provenance",
        "version": 2,
        "created_at_utc": lsd_runner.utc_now(),
        "variant": "D_drift_lsd_hybrid",
        "loss": "L_mixed_LSD + lambda_GMD * L_official_GMD(X(x0,0,1),data)",
        "selected_gmd_lambda": float(args.gmd_lambda),
        "lambda_origin": (
            "equal_RMS_gradient_calibration"
            if calibration is not None and args._lambda_derived_in_this_run
            else "explicit_fixed_cli"
        ),
        "gradient_calibration": calibration,
        "feature_encoder": dict(feature_metadata),
        "hybrid_source_snapshot": snapshot,
        "comparison_contract": comparison,
        "run_role": (
            "calibration_smoke"
            if args.calibration_smoke
            else "resumed_fixed_lambda" if args.resume is not None else "scientific_fixed_lambda"
        ),
        "budget_updates": args.steps,
    }
    lsd_runner.atomic_json_dump(run_dir / f"hybrid_provenance{suffix}.json", payload)


def train(args: argparse.Namespace) -> Path:
    """Prepare calibration/geometry, then reuse the pure-LSD train engine.

    The three temporary hooks are process-local and restored in ``finally``.
    They alter only model construction, signature enrichment, and metadata;
    the attested baseline remains authoritative for its stream, RAdam, EMA,
    exact current-target LSD forward, DDP wrapping, clipping, checkpoints,
    resume semantics, and exports.
    """

    cache_latents, cache_metadata = lsd_runner.load_latent_cache(args.cache)
    selected_spatial, training_latents, _, _ = (
        lsd_runner._select_training_latents(cache_latents, args)
    )
    cache_digest = lsd_runner.sha256_file(args.cache)
    # Match the baseline signature byte-for-byte: its attested digest includes
    # the selected tensor's spatial [N,4,8,8] shape, not the flattened view.
    train_values_digest = lsd_runner.sha256_tensor(selected_spatial)
    validate_baseline_a_data_identity(
        args,
        cache_sha256=cache_digest,
        train_values_sha256=train_values_digest,
    )
    feature_encoder, feature_metadata = load_feature_encoder(
        args, cache_metadata, cache_digest
    )
    calibration: dict[str, Any] | None = None
    args._lambda_derived_in_this_run = False
    if args.gradient_calibration_batches:
        # Construct an expendable theta_0 solely for autograd.grad.  The base
        # engine reseeds and reconstructs the scientific model afterwards, so
        # calibration cannot perturb initialization, data position, or RNG.
        lsd_runner.seed_initial_parameters(args)
        calibration_model = make_hybrid_model(
            args,
            feature_encoder,
            gmd_lambda=1.0 if args.gmd_lambda is None else args.gmd_lambda,
        ).to(args.device)
        torch.manual_seed(args.seed + args.rank)
        if args.device.type == "cuda":
            torch.cuda.manual_seed(args.seed + args.rank)
        calibration = calibrate_lambda(calibration_model, training_latents, args)
        if args.gmd_lambda is None:
            args.gmd_lambda = float(calibration["lambda_equal_rms"])
            args._lambda_derived_in_this_run = True
        del calibration_model
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    assert args.gmd_lambda is not None

    original_make_model = lsd_runner.make_training_model
    original_signature = lsd_runner.training_signature
    original_write_metadata = lsd_runner.write_metadata

    def make_model_hook(hook_args: argparse.Namespace) -> DriftLSDHybridTrainingModel:
        pure_model = original_make_model(hook_args)
        return DriftLSDHybridTrainingModel(
            pure_model.flow_map,
            feature_encoder,
            gmd_lambda=float(hook_args.gmd_lambda),
            temperatures=hook_args.gmd_temperatures,
            distributed_statistics=hook_args.distributed,
        )

    def signature_hook(*positional: Any, **keywords: Any) -> dict[str, Any]:
        base = original_signature(*positional, **keywords)
        return augment_hybrid_training_signature(
            base,
            args,
            feature_metadata,
        )

    def write_metadata_hook(
        hook_args: argparse.Namespace,
        run_dir: Path,
        *positional: Any,
        **keywords: Any,
    ) -> None:
        original_write_metadata(
            hook_args,
            run_dir,
            *positional,
            **keywords,
        )
        write_hybrid_metadata(
            hook_args,
            run_dir,
            feature_metadata,
            calibration,
        )
        if calibration is not None:
            lsd_runner.JsonlLogger(run_dir / "metrics.jsonl").log(
                {
                    "event": "gradient_calibration",
                    "selected_gmd_lambda": float(hook_args.gmd_lambda),
                    "lambda_derived_in_this_run": hook_args._lambda_derived_in_this_run,
                    **calibration,
                }
            )

    lsd_runner.make_training_model = make_model_hook
    lsd_runner.training_signature = signature_hook
    lsd_runner.write_metadata = write_metadata_hook
    try:
        return lsd_runner.train(args)
    finally:
        lsd_runner.make_training_model = original_make_model
        lsd_runner.training_signature = original_signature
        lsd_runner.write_metadata = original_write_metadata


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    try:
        lsd_runner.setup_distributed(args)
        validate_hybrid_args(args)
        return train(args)
    finally:
        lsd_runner.cleanup_distributed(args)


if __name__ == "__main__":
    main()
