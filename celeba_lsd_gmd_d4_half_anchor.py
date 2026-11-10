#!/usr/bin/env python3
"""Run the pre-registered D4 half-strength CelebA LSD/GMD ablation.

D4 deliberately reuses the audited D2 training implementation without
modifying it.  Before argument parsing, this thin fail-closed adapter replaces
only the registered experiment identity, run directory, protocol pin and GMD
lambda target.  The actual objective, optimizer, data stream, exact-fork
loader, EMA and checkpoint machinery remain D2's tested code paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import shutil
from typing import Any

import celeba_lsd_gmd_ramped_anchor as d2
from local_paths import REPO_ROOT


OBJECTIVE = "mixed_lsd_plus_half_strength_linearly_ramped_terminal_official_gmd_mae_v2"
INTERVENTION_ID = "D4_half_strength_ramped_terminal_GMD"
TARGET_FRACTION_OF_EQUAL_RMS = 0.05
LAMBDA_TARGET = 0.00012095788737580082
EXPECTED_RUN_DIR = (
    REPO_ROOT
    / "runs/celeba_latent_lsd/"
    "lsd_independent_std1_base256_seed42_lr003_ramped_gmd5pct_40k_d4_v1"
)
PROTOCOL_PATH = REPO_ROOT / "CELEBA_LSD_GMD_D4_HALF_ANCHOR_PROTOCOL.md"
PROTOCOL_SHA256 = "7054baa5e1effe087dc2ae82fb8bdfb57244002be2bc2f41587eba969e6b05b0"

D4_SOURCE_FILES = (
    'celeba_lsd_gmd_d4_half_anchor.py',
    'celeba_lsd_gmd_ramped_anchor.py',
    'celeba_latent_lsd.py',
    'celeba_drift_lsd_hybrid.py',
    'src/celeba_drift_lsd_hybrid.py',
    'src/celeba_gmd_features_v2.py',
    'src/official_drifting.py',
    'src/celeba_latent_lsd.py',
    'src/flow_map_self_distillation.py',
    'src/celeba_latent.py',
    'local_paths.py',
    'pyproject.toml',
)

_D2_BUILD_PARSER = d2.build_parser
_D2_AUGMENT_SIGNATURE = d2.augment_signature


def build_parser() -> argparse.ArgumentParser:
    """Return the D2 parser with only D4 experiment defaults changed."""

    parser = _D2_BUILD_PARSER()
    parser.description = "CelebA LR=.003 LSD continuation with half-strength ramped GMD"
    parser.set_defaults(
        output_root=REPO_ROOT / "runs/celeba_latent_lsd",
        run_label="lsd_lr003_ramped_gmd5pct_40k_d4",
        steps=d2.RAMP_END_STEP,
        lr=0.003,
        checkpoint_every=2_500,
        log_every=10,
        gmd_lambda_target=LAMBDA_TARGET,
    )
    return parser


def augment_signature(
    base: Mapping[str, Any],
    args: argparse.Namespace,
    feature_metadata: Mapping[str, Any],
    schedule: d2.LinearLambdaSchedule,
) -> dict[str, Any]:
    result = _D2_AUGMENT_SIGNATURE(base, args, feature_metadata, schedule)
    result.update(
        {
            "objective": OBJECTIVE,
            "intervention_id": INTERVENTION_ID,
            "terminal_gmd_target_fraction_of_equal_rms": (
                TARGET_FRACTION_OF_EQUAL_RMS
            ),
            "d4_only_change_from_d2": (
                "terminal_gmd_target_fraction_of_equal_rms:0.10_to_0.05"
            ),
            "d4_source_files_sha256": d2.anchor_source_hashes(),
        }
    )
    return result


def archive_d4_sources(run_dir: Path, suffix: str) -> dict[str, Any]:
    destination = run_dir / f"d4_half_anchor_source_snapshot{suffix}"
    destination.mkdir(parents=True, exist_ok=False)
    hashes = d2.anchor_source_hashes()
    for name in D4_SOURCE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
    payload = {"created_at_utc": d2.lsd_runner.utc_now(), "source_files_sha256": hashes}
    d2.lsd_runner.atomic_json_dump(destination / "manifest.json", payload)
    return {"path": str(destination), **payload}


def write_d4_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    feature_metadata: Mapping[str, Any],
    schedule: d2.LinearLambdaSchedule,
    *,
    initial_fork: bool,
) -> None:
    suffix = (
        "_fork_" if initial_fork else "_resume_"
    ) + d2.lsd_runner.utc_now().replace(":", "-")
    snapshot = archive_d4_sources(run_dir, suffix)
    payload = {
        "format": "driftmodel.celeba_lsd_gmd_d4_half_anchor_provenance",
        "version": 1,
        "created_at_utc": d2.lsd_runner.utc_now(),
        "objective": OBJECTIVE,
        "intervention_id": INTERVENTION_ID,
        "run_role": "initial_exact_fork" if initial_fork else "exact_resume",
        "schedule": schedule.as_dict(),
        "feature_encoder": dict(feature_metadata),
        "pure_fork": {
            "path": str(args.fork_from_pure_checkpoint.resolve())
            if args.fork_from_pure_checkpoint is not None
            else str(d2.PURE_CHECKPOINT),
            "sha256": args.fork_checkpoint_sha256 or d2.PURE_CHECKPOINT_SHA256,
            "completed_step": schedule.start_step,
            "restored_state": [
                "raw_model",
                "RAdam",
                "EMA",
                "data_stream",
                "rng_by_rank",
            ],
            "optimizer_reset": False,
        },
        "pre_registered_protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": PROTOCOL_SHA256,
        },
        "gradient_diagnostic": {
            "path": str(d2.DIAGNOSTIC_REPORT),
            "sha256": d2.DIAGNOSTIC_REPORT_SHA256,
            "lambda_equal_rms_raw_30k": d2.LAMBDA_EQUAL_RMS_RAW_30K,
            "chosen_fraction": TARGET_FRACTION_OF_EQUAL_RMS,
            "lambda_target": LAMBDA_TARGET,
        },
        "controlled_ablation": {
            "reference": "D2_ramped_GMD_10pct",
            "only_change": "lambda_target_halved",
            "reference_fraction": 0.1,
            "d4_fraction": TARGET_FRACTION_OF_EQUAL_RMS,
        },
        "source_snapshot": snapshot,
        "source_manifest": {
            "path": str(args.expected_source_manifest.resolve())
            if args.expected_source_manifest is not None
            else None,
            "sha256": args.expected_source_manifest_sha256,
        },
        "claim_limit": (
            "single-seed pre-registered dose ablation; not a paper result"
        ),
    }
    d2.lsd_runner.atomic_json_dump(
        run_dir / f"d4_half_anchor_provenance{suffix}.json", payload
    )


def configure_d4() -> None:
    """Install the D4 constants/hooks in the imported D2 engine."""

    d2.OBJECTIVE = OBJECTIVE
    d2.LAMBDA_TARGET = LAMBDA_TARGET
    d2.EXPECTED_RUN_DIR = EXPECTED_RUN_DIR
    d2.PROTOCOL_PATH = PROTOCOL_PATH
    d2.PROTOCOL_SHA256 = PROTOCOL_SHA256
    d2.ANCHOR_SOURCE_FILES = D4_SOURCE_FILES
    d2.build_parser = build_parser
    d2.augment_signature = augment_signature
    d2.write_anchor_metadata = write_d4_metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    configure_d4()
    return d2.parse_args(argv)


def validate_d4_args(args: argparse.Namespace) -> d2.LinearLambdaSchedule:
    configure_d4()
    return d2.validate_anchor_args(args)


def main(argv: list[str] | None = None) -> Path:
    configure_d4()
    return d2.main(argv)


if __name__ == "__main__":
    main()
