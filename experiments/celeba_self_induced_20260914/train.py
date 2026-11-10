#!/usr/bin/env python3
"""Fresh self-induced Flow Maps of Drifting; factorial loss ablations."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_rank_cache() -> None:
    base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
    if not base:
        return
    root = Path(base) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    for directory in (root, root / "userdb", root / "kernels"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        MIOPEN_CACHE_DIR=str(root),
        MIOPEN_USER_DB_PATH=str(root / "userdb"),
        MIOPEN_CUSTOM_CACHE_DIR=str(root / "kernels"),
    )


_configure_rank_cache()

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange

import celeba_pixel_gmd_generator_train as base
from celeba_pixel_gmd_mae_train import (
    CelebA64PixelDataset,
    JsonlLogger,
    atomic_json_dump,
    atomic_text_write,
    atomic_torch_save,
    build_dataset_manifest,
    torch_tree_sha256,
)
from experiments.celeba_baseline_20260911 import gmd_train as official
from src.celeba_gmd_composed_rollout import (
    ComposedRollout,
    composed_rollout_artifact_dict,
    load_composed_rollout_artifact,
    unwrap_composed_rollout,
)
from src.celeba_gmd_sharded_features import global_sharded_mae_convnext_loss
from src.celeba_pixel_gmd_convnext import load_local_convnextv2_base
from src.celeba_pixel_gmd_features import (
    canonical_json_sha256,
    load_frozen_pixel_feature_encoder,
    sha256_file,
    state_dict_sha256,
)
from src.celeba_pixel_gmd_generator import NegativeImageBank, PositiveImageBank
from src.celeba_pixel_gmd_mae_convnext import (
    FrozenMAEConvNextFeatures,
    composite_representation_signature,
)


from src.celeba_self_induced import FACTORS, SignalWeights, SelfInducedSignals, masked_weights, sample_times, signal_objective
from src.celeba_gmd_flow_map import CelebA64TwoTimeGMDDiT

TRAINING_FORMAT = "driftmodel.celeba64_self_induced_training"
TRAINING_VERSION = 1
SIGNATURE_VERSION = 1
TRAIN_FILES_SHA256 = official.TRAIN_FILES_SHA256
SOURCE_FILES = (
    'experiments/celeba_self_induced_20260914/train.py',
    'src/celeba_self_induced.py',
    'src/celeba_gmd_local_two_view.py',
    'experiments/celeba_baseline_20260911/gmd_train.py',
    'src/celeba_gmd_flow_map.py',
    'src/celeba_gmd_flow_objectives.py',
    'src/celeba_gmd_composed_rollout.py',
    'src/celeba_gmd_sharded_features.py',
    'celeba_pixel_gmd_generator_train.py',
    'celeba_pixel_gmd_mae_train.py',
    'src/celeba_pixel_gmd_generator.py',
    'src/celeba_pixel_gmd_features.py',
    'src/official_drifting.py',
    'src/celeba_pixel_gmd_convnext.py',
    'src/celeba_pixel_gmd_mae_convnext.py',
)
SOURCE_FILES += ()


def build_parser():
    parser = official.build_parser()
    parser.description = __doc__
    parser.set_defaults(steps=5000, checkpoint_every=500, log_every=20,
                        output_root=ROOT / "runs/celeba_self_induced_20260914/training",
                        lr=2e-4, lr_warmup_steps=1000, deterministic=True)
    parser.add_argument("--mask", default="111111")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--time-scale", type=float, default=10.0)
    parser.add_argument("--time-seed", type=int, default=73001)
    parser.add_argument("--distill-max-span", type=float, default=0.5)
    parser.add_argument("--training-data-control", choices=("none", "pixel_permutation"), default="none")
    for name, value in SignalWeights().as_dict().items():
        parser.add_argument("--weight-" + name.replace("_", "-"), type=float, default=value)
    return parser


def weights_from_args(args):
    return masked_weights(args.mask, SignalWeights(**{k: getattr(args, "weight_" + k) for k in FACTORS}))


def validate_args(args):
    official.validate_official_inputs(args)
    weights_from_args(args)
    if args.world_size != 8 or args.device.type != "cuda":
        raise ValueError("production campaign requires eight ROCm ranks")
    if not 0 < args.dt < 0.5 or not 0 < args.dt * args.time_scale <= 1:
        raise ValueError("require 0<dt<0.5 and 0<S*dt<=1")
    if not 0 < args.distill_max_span <= 1:
        raise ValueError("invalid distillation span")
    if (args.global_generated, args.positive_count, args.negative_count, args.bank_push) != (64, 128, 32, 128):
        raise ValueError("MAE/ConvNeXt support contract must be 64/128/32, push128")
    if args.augmentation != "aligned_flip" or not args.preload_uint8 or not args.use_bfloat16:
        raise ValueError("aligned flip, uint8 preload, BF16 are required")
    if tuple(args.temperatures) != (0.2, 0.05, 0.02):
        raise ValueError("official GMD temperatures must remain unchanged")
    if args.steps < 1 or args.log_every < 1 or args.checkpoint_every < 0:
        raise ValueError("invalid update/log/checkpoint count")
    if not 0 < args.lr or not 0 <= args.lr_warmup_start <= args.lr or args.lr_warmup_steps < 0:
        raise ValueError("invalid learning rate")
    if args.resume is not None and not _is_sha(args.resume_sha256):
        raise ValueError("resume requires an explicit checkpoint SHA-256")
    args.local_generated = 8
    # The launcher fixes source hashes before spawning any ranks.
    manifest = json.loads(args.expected_source_manifest.read_text())
    if sha256_file(args.expected_source_manifest) != args.expected_source_manifest_sha256:
        raise ValueError("launcher manifest hash differs")
    if manifest != _source_hashes():
        raise ValueError("source changed after submission")
    args.launcher_source_hashes = manifest


def new_flow(args):
    """Random initialization only: no pretrained generator is opened."""
    return CelebA64TwoTimeGMDDiT(
        patch_size=args.patch_size, hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        noise_classes=args.noise_classes, noise_coords=args.noise_coords,
        condition_tokens=args.condition_tokens, gradient_checkpointing=args.gradient_checkpointing,
        intended_inference_precision="bfloat16_autocast",
    )


def training_signature(args, dataset_manifest, flow, feature_identity, feature_encoder, initial_hash):
    return {
        "version": SIGNATURE_VERSION, "experiment": "self_induced_factorial_20260914",
        "objective": "self_induced_two_view_feature_drifting_plus_independent_L_E_S_B0_Ba",
        "flow": flow.flow_architecture_config(), "mask": args.mask,
        "weights": weights_from_args(args).as_dict(),
        "initialization": {"kind": "random", "generator_checkpoint": None,
                           "seed": args.seed, "completed_step": 0, "model_state_sha256": initial_hash},
        "dataset": {k: v for k, v in dataset_manifest.items() if k != "files"},
        "training_data_control": args.training_data_control,
        "feature_checkpoint": feature_identity,
        "feature_representation": composite_representation_signature(feature_encoder),
        "time": {"dt": args.dt, "S": args.time_scale, "force_step": args.dt * args.time_scale,
                 "max_span": args.distill_max_span, "seed": args.time_seed,
                 "groups": 1, "support": "one homogeneous q_a globally",
                 "rng": "three fixed draws per update, independent of enabled losses"},
        "terminal_training": "one direct boundary transition; intermediate source detached; no terminal rollout",
        "lagrangian": "finite displacement residual as MNIST fast_wgf_flow_map_loss, no JVP",
        "supports": {"generated_global": 64, "positive": 128, "fixed_negative": 32,
                     "positive_bank": args.positive_bank_size, "negative_bank": args.negative_bank_size},
        "temperatures": list(args.temperatures),
        "optimizer": {"name": "AdamW", "lr": args.lr, "warmup_start": args.lr_warmup_start,
                      "warmup_steps": args.lr_warmup_steps, "betas": [args.adam_beta1, args.adam_beta2],
                      "eps": args.adam_eps, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip},
        "ema_decay": args.ema_decay, "world_size": args.world_size,
        "precision": "FP32 weights/optimizer; BF16 generator/MAE; FP32 ConvNeXt and GMD; autocast cache disabled",
        "source_files_sha256": dict(args.launcher_source_hashes),
    }

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    invoked = sys.argv[1:] if argv is None else argv
    args.command = shlex.join([sys.argv[0], *invoked])
    return args

def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _source_hashes() -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required Flow-GMD sources are absent: {missing}")
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}

def setup_distributed(args: argparse.Namespace) -> None:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("Flow-GMD training requires ROCm GPUs")
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    if args.distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=args.distributed_timeout_minutes))
    validate_args(args)

def cleanup(args: argparse.Namespace) -> None:
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()

def barrier(args: argparse.Namespace) -> None:
    if args.distributed:
        dist.barrier(device_ids=[args.local_rank])

def rank_zero(args: argparse.Namespace, operation):
    barrier(args)
    result = None
    failure = None
    if args.rank == 0:
        try:
            result = operation()
        except Exception:
            failure = traceback.format_exc()
    if args.distributed:
        payload = [failure, result]
        dist.broadcast_object_list(payload, src=0)
        failure, result = payload
    if failure:
        raise RuntimeError(f"rank-zero operation failed:\n{failure}")
    barrier(args)
    return result

def gather(local: Any, args: argparse.Namespace) -> list[Any]:
    if not args.distributed:
        return [local]
    values: list[Any] = [None] * args.world_size
    dist.all_gather_object(values, local)
    return values

def resolve_run_dir(args: argparse.Namespace) -> Path:
    def resolve() -> str:
        if args.resume is not None:
            parent = args.resume.resolve().parent
            inferred = parent.parent if parent.name == "checkpoints" else parent
            if args.run_dir is not None and args.run_dir.resolve() != inferred:
                raise ValueError("run directory differs from resume checkpoint")
            if not inferred.is_dir():
                raise FileNotFoundError("resume run directory is absent")
            return str(inferred)
        if args.run_dir is not None:
            path = args.run_dir.resolve()
            if path.exists() and any(path.iterdir()):
                raise FileExistsError("new run directory is not empty")
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        label = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.run_label).strip("_")
        path = (args.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{label}").resolve()
        path.mkdir(parents=True, exist_ok=False)
        return str(path)
    return Path(rank_zero(args, resolve))

def _snapshot(run_dir: Path, expected: Mapping[str, str], name: str) -> dict[str, Any]:
    destination = run_dir / name
    if destination.exists():
        raise FileExistsError(destination)
    before = _source_hashes()
    if before != dict(expected):
        raise RuntimeError("source snapshot differs from launcher manifest")
    for relative in SOURCE_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    copied = {relative: sha256_file(destination / relative) for relative in SOURCE_FILES}
    if copied != before or _source_hashes() != before:
        raise RuntimeError("sources changed while snapshotting")
    manifest = {"format": "driftmodel.celeba64_flow_gmd_source_snapshot", "version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": copied}
    path = atomic_json_dump(destination / "manifest.json", manifest)
    return {**manifest, "manifest": str(path.resolve()), "manifest_sha256": sha256_file(path)}

def write_metadata(args, run_dir, dataset_manifest, signature, flow) -> None:
    suffix = "" if args.resume is None else f"_resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot = _snapshot(run_dir, args.launcher_source_hashes, f"source_snapshot{suffix}")
    atomic_json_dump(run_dir / f"dataset_manifest{suffix}.json", dataset_manifest)
    config = {key: (str(value) if isinstance(value, (Path, torch.device)) else value) for key, value in vars(args).items() if not key.startswith("_")}
    config.update(parameter_count=sum(parameter.numel() for parameter in flow.parameters()), training_signature=signature)
    atomic_json_dump(run_dir / f"config{suffix}.json", json.loads(json.dumps(config, default=str)))
    atomic_json_dump(run_dir / f"provenance{suffix}.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "command": args.command, "hostname": socket.gethostname(), "torch": str(torch.__version__), "hip": getattr(torch.version, "hip", None), "world_size": args.world_size, "source_snapshot": snapshot, "parent": signature["initialization"]})

def _checkpoint_integrity(payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "model_state_sha256": state_dict_sha256(payload["model"]),
        "ema_state_sha256": state_dict_sha256(payload["ema"]),
        "optimizer_state_sha256": torch_tree_sha256(payload["optimizer"]),
        "stream_state_sha256": torch_tree_sha256(payload["data_stream"]),
        "positive_bank_state_sha256": torch_tree_sha256(payload["positive_bank"]),
        "negative_bank_state_sha256": torch_tree_sha256(payload["negative_bank"]),
        "bank_rng_state_sha256": torch_tree_sha256(payload["bank_rng_state"]),
        "prior_rng_states_sha256": torch_tree_sha256(payload["prior_rng_by_rank"]),
        "time_rng_state_sha256": torch_tree_sha256(payload["time_rng_state"]),
        "time_counts_sha256": torch_tree_sha256(payload["time_counts"]),
        "training_signature_sha256": canonical_json_sha256(dict(payload["training_signature"])),
    }

def save_checkpoint(path, *, completed_step, model, ema, optimizer, stream, positive_bank, negative_bank, bank_generator, prior_generator, time_generator, time_counts, signature, run_dir, args) -> str:
    prior_states = gather(prior_generator.get_state().cpu(), args)
    globals_by_rank = gather({"stream": stream.state_dict(), "positive": positive_bank.state_dict(), "negative": negative_bank.state_dict(), "bank_rng": bank_generator.get_state().cpu(), "time_rng": time_generator.get_state().cpu(), "time_counts": dict(time_counts)}, args)
    global_hashes = [torch_tree_sha256(value) for value in globals_by_rank]
    if len(set(global_hashes)) != 1:
        raise RuntimeError("global stream/banks/K RNG diverged across ranks")
    shared = globals_by_rank[0]

    def save() -> str:
        flow = unwrap_composed_rollout(model).flow
        payload = {
            "format": TRAINING_FORMAT, "version": TRAINING_VERSION, "completed_step": completed_step,
            "model": {name: value.detach().cpu() for name, value in flow.state_dict().items()},
            "ema": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
            "optimizer": optimizer.state_dict(), "data_stream": shared["stream"], "positive_bank": shared["positive"], "negative_bank": shared["negative"], "bank_rng_state": shared["bank_rng"], "prior_rng_by_rank": prior_states, "time_rng_state": shared["time_rng"], "time_counts": shared["time_counts"], "training_signature": dict(signature), "run_dir": str(run_dir.resolve()), "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload["integrity"] = _checkpoint_integrity(payload)
        atomic_torch_save(path, payload)
        digest = sha256_file(path)
        atomic_text_write(Path(f"{path}.sha256"), f"{digest}  {path.name}\n")
        return digest
    return str(rank_zero(args, save))

def load_checkpoint(path, expected_sha, *, model, ema, optimizer, stream, positive_bank, negative_bank, bank_generator, prior_generator, time_generator, time_counts, signature, run_dir, args) -> int:
    if sha256_file(path) != expected_sha:
        raise ValueError("resume external SHA-256 differs")
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != f"{expected_sha}  {path.name}":
        raise ValueError("resume sidecar differs")
    payload = torch.load(path, map_location=args.device, weights_only=True)
    expected = {"format", "version", "completed_step", "model", "ema", "optimizer", "data_stream", "positive_bank", "negative_bank", "bank_rng_state", "prior_rng_by_rank", "time_rng_state", "time_counts", "training_signature", "run_dir", "saved_at_utc", "integrity"}
    if not isinstance(payload, Mapping) or set(payload) != expected or payload["format"] != TRAINING_FORMAT or payload["version"] != TRAINING_VERSION:
        raise ValueError("resume Flow-GMD schema differs")
    if dict(payload["integrity"]) != _checkpoint_integrity(payload):
        raise ValueError("resume internal integrity differs")
    if payload["training_signature"] != dict(signature) or Path(payload["run_dir"]).resolve() != run_dir.resolve():
        raise ValueError("resume signature or run directory differs")
    unwrap_composed_rollout(model).flow.load_state_dict(payload["model"], strict=True)
    ema.load_state_dict(payload["ema"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    stream.load_state_dict(payload["data_stream"])
    positive_bank.load_state_dict(payload["positive_bank"])
    negative_bank.load_state_dict(payload["negative_bank"])
    bank_generator.set_state(payload["bank_rng_state"].cpu())
    prior_generator.set_state(payload["prior_rng_by_rank"][args.rank].cpu())
    time_generator.set_state(payload["time_rng_state"].cpu())
    time_counts.clear(); time_counts.update({str(k): int(v) for k, v in payload["time_counts"].items()})
    return int(payload["completed_step"])

def export_artifacts(run_dir, completed_step, model, ema, signature, checkpoint, checkpoint_sha) -> None:
    def metadata(kind):
        return {"completed_step": completed_step, "weight_kind": kind, "training_signature": dict(signature), "training_signature_sha256": canonical_json_sha256(dict(signature)), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    raw_path, ema_path = run_dir / "flow_raw_final.pt", run_dir / "flow_ema_final.pt"
    atomic_torch_save(raw_path, composed_rollout_artifact_dict(model, metadata=metadata("raw")))
    atomic_torch_save(ema_path, composed_rollout_artifact_dict(ComposedRollout(ema), metadata=metadata("ema")))
    load_composed_rollout_artifact(raw_path, "cpu")
    load_composed_rollout_artifact(ema_path, "cpu")
    manifest = {"format": "driftmodel.celeba64_flow_gmd_artifacts", "version": 1, "completed_step": completed_step, "strict_reload": True, "artifacts": {"training_checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha}, "raw": {"path": str(raw_path.resolve()), "sha256": sha256_file(raw_path)}, "ema": {"path": str(ema_path.resolve()), "sha256": sha256_file(ema_path)}}}
    path = atomic_json_dump(run_dir / "artifact_manifest.json", manifest)
    atomic_text_write(run_dir / "artifact_manifest.json.sha256", f"{sha256_file(path)}  {path.name}\n")


def train(args):
    run_dir = resolve_run_dir(args)
    dataset_manifest = rank_zero(args, lambda: build_dataset_manifest(args.data_dir.resolve(), 0, 162770))
    if dataset_manifest["files_sha256"] != TRAIN_FILES_SHA256 or dataset_manifest["image_count"] != 162770:
        raise ValueError("official training split differs")
    dataset = CelebA64PixelDataset(args.data_dir.resolve(), dataset_manifest, preload_uint8=True)
    stream = base.GlobalBankStream(dataset, batch_size=128, seed=args.seed + 101)
    positive_bank = PositiveImageBank(args.positive_bank_size)
    negative_bank = NegativeImageBank(args.negative_bank_size)
    bank_generator = base.make_bank_generator(args)
    feature_identity = base.feature_checkpoint_identity(args)
    mae = load_frozen_pixel_feature_encoder(args.feature_checkpoint, args.device)
    convnext = load_local_convnextv2_base(args.convnext_snapshot_dir, device=args.device)
    encoder = FrozenMAEConvNextFeatures(mae, convnext).eval()
    base.seed_parameters(args)
    flow = new_flow(args).to(args.device)
    initial_hash = rank_zero(args, lambda: state_dict_sha256(flow.state_dict()))
    ema = copy.deepcopy(flow).eval().requires_grad_(False)
    weights = weights_from_args(args)
    wrapper = SelfInducedSignals(flow, weights, dt=args.dt).train()
    model = DistributedDataParallel(wrapper, device_ids=[args.local_rank], output_device=args.local_rank,
                                    broadcast_buffers=False, find_unused_parameters=False, static_graph=False)
    prior_generator = base.make_prior_generator(args)
    time_generator = torch.Generator(device="cpu").manual_seed(args.time_seed)
    # Fixed permutation, using its own RNG; no changes to the bank/prior streams.
    permutation = torch.randperm(64 * 64, generator=torch.Generator().manual_seed(97531))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                                 weight_decay=args.weight_decay)
    signature = training_signature(args, dataset_manifest, flow, feature_identity, encoder, initial_hash)
    checkpoint_args = dict(model=model, ema=ema, optimizer=optimizer, stream=stream,
                           positive_bank=positive_bank, negative_bank=negative_bank,
                           bank_generator=bank_generator, prior_generator=prior_generator,
                           time_generator=time_generator, time_counts={"updates": 0},
                           signature=signature, run_dir=run_dir, args=args)
    start = 0
    if args.resume is not None:
        start = load_checkpoint(args.resume, args.resume_sha256, **checkpoint_args)
    if start >= args.steps:
        raise ValueError("target must exceed resumed step")
    rank_zero(args, lambda: write_metadata(args, run_dir, dataset_manifest, signature, flow))
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    if args.rank == 0:
        logger.log({"event": "resume" if start else "start", "step": start,
                    "mask": args.mask, "weights": weights.as_dict(), "initialization": signature["initialization"],
                    "world_size": args.world_size, "job_id": os.environ.get("SLURM_JOB_ID")})
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.monotonic()
    for index in range(start, args.steps):
        step = index + 1
        tick = time.monotonic()
        lr = base.learning_rate_at_step(index, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        pushed = base.augment_bank_images(stream.next(), "aligned_flip", generator=bank_generator)
        if args.training_data_control == "pixel_permutation":
            pushed = pushed.flatten(2)[:, :, permutation].reshape_as(pushed)
        positive_bank.add(pushed)
        negative_bank.add(pushed)
        positives = positive_bank.sample(128, generator=bank_generator).to(args.device, non_blocking=True)
        negatives = negative_bank.sample(32, generator=bank_generator).to(args.device, non_blocking=True)
        prior, labels = flow.sample_prior(args.local_generated, device=args.device, generator=prior_generator)
        times = sample_times(time_generator, dt=args.dt, max_span=args.distill_max_span)
        times_tensor = torch.tensor(times, dtype=torch.float64, device=args.device)
        broadcast_times = times_tensor.clone()
        dist.broadcast(broadcast_times, src=0)
        if not torch.equal(times_tensor, broadcast_times):
            raise RuntimeError("time RNG diverged across ranks")
        checkpoint_args["time_counts"]["updates"] += 1
        optimizer.zero_grad(set_to_none=True)
        # Disable the shared parameter cast cache across teacher and student
        # forwards, including non-reentrant checkpoint recomputation.
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            outputs = model(prior, labels, *times)
            loss, components = signal_objective(
                outputs, weights, encoder, positives, negatives,
                force_step=args.dt * args.time_scale, temperatures=args.temperatures,
            )
        finite = torch.isfinite(loss.detach()).int()
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"nonfinite objective at step {step}")
        loss.backward()
        if step == start + 1:
            if any(p.grad is not None for p in encoder.parameters()):
                raise RuntimeError("frozen encoder received parameter gradients")
            if any(p.grad is None for p in model.parameters()):
                raise RuntimeError("flow parameter disconnected from the active objective")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        base.update_ema(ema, flow, args.ema_decay)
        encoder.clear_step_state()
        if step == start + 1 or step == args.steps or step % args.log_every == 0:
            torch.cuda.synchronize(args.device)
            values = {"total_loss": loss.detach(), "gradient_norm": gradient_norm,
                      **{"loss/" + k: v.detach() for k, v in components.items()}}
            values = base.reduce_stats(values, args)
            maxima = base.reduce_max_stats({"seconds_per_update": time.monotonic() - tick,
                       "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                       "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)}, args)
            if args.rank == 0:
                record = {"event": "train", "step": step, "mask": args.mask,
                          "times": list(times), "lr": lr, "elapsed_seconds": time.monotonic() - started,
                          **{k: float(v.cpu()) for k, v in {**values, **maxima}.items()}}
                logger.log(record)
                print(json.dumps(record), flush=True)
        del loss, components, outputs
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step != args.steps:
            path = run_dir / "checkpoints" / f"checkpoint_step_{step:08d}.pt"
            digest = save_checkpoint(path, completed_step=step, **checkpoint_args)
            if args.rank == 0:
                logger.log({"event": "checkpoint", "step": step, "sha256": digest, "path": str(path)})
    path = run_dir / "checkpoint_final.pt"
    digest = save_checkpoint(path, completed_step=args.steps, **checkpoint_args)
    rank_zero(args, lambda: export_artifacts(run_dir, args.steps, model, ema, signature, path, digest))
    if args.rank == 0:
        logger.log({"event": "complete", "step": args.steps, "checkpoint_sha256": digest,
                    "elapsed_seconds": time.monotonic() - started})
    return run_dir


def main(argv=None):
    import fcntl
    args = parse_args(argv)
    # One producer per run, including across Slurm retries. The lock is kept
    # by rank zero throughout training and is released on process termination.
    lock = None
    try:
        setup_distributed(args)
        if args.run_dir is None:
            raise ValueError("an explicit run directory is required")
        if args.rank == 0:
            args.run_dir.parent.mkdir(parents=True, exist_ok=True)
            lock = open(args.run_dir.parent / (args.run_dir.name + ".lock"), "a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return train(args)
    finally:
        if lock is not None:
            lock.close()
        cleanup(args)


if __name__ == "__main__":
    main()
