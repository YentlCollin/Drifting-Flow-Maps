from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import trange

from local_paths import REPO_ROOT
from src.ddim import ddim, diffusion_loss, mk_schedule, model_denoiser, q_sample
from src.model import (
    MNIST_DIM,
    MnistConvGenerator,
    TimeConditionedMnistConvGenerator,
    TimeConditionedMnistDenoiser,
    TransitionConditionedMnistConvGenerator,
)
from src.utils import _next_batch, _save_mnist_grid, load_mnist_dataset
from src.wgf import (
    evaluate_flow_map,
    euler_flow_matching_loss,
    fast_wgf_flow_map_loss,
    optimizer_time_drifting_loss,
    sample_wgf_flow_map,
    wgf_flow_map_loss,
)


PLOT_TIMES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
WGF_FLOW_MAP_SAMPLE_STEPS = [1, 2, 4, 8, 16, 50]
DDIM_SCHEDULE = {"schedule_type": "vp", "beta_min": 1e-4, "beta_max": 2e-2}


def run_dir(method: str, label: str = ""):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
    run_name = f"{stamp}_{safe_label}" if safe_label else stamp
    path = REPO_ROOT / "runs" / "mnist_wgf" / method / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_float_dict(stats: dict) -> dict:
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        for key, value in stats.items()
    }


def write_history(path, history: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_history(path, history: list[dict], keys: list[str], title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for key in keys:
        plt.plot([row[key] for row in history], label=key)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_config(path, args, method: str):
    config = vars(args).copy()
    config["method"] = method
    if method != "ddim":
        config.pop("ddim_steps", None)
        config.pop("ddim_eta", None)
    if method == "fast_wgf_flow_map":
        for key in (
            "terminal_loss",
            "lambda_sg",
            "psd_weight",
            "lambda_terminal",
            "terminal_steps",
            "distill_min_span",
            "path_weight",
        ):
            config.pop(key)
        active_losses = []
        if args.fast_path_weight:
            if args.fast_lambda_wgf:
                active_losses.append("wgf_origin_and_local")
            if args.fast_lambda_lagrangian:
                active_losses.append("lagrangian_fd")
            if args.fast_lambda_eulerian:
                active_losses.append("eulerian_fd")
        boundary_origin_weight = (
            args.fast_lambda_boundary
            if args.fast_lambda_boundary_origin is None
            else args.fast_lambda_boundary_origin
        )
        boundary_arbitrary_weight = (
            args.fast_lambda_boundary
            if args.fast_lambda_boundary_arbitrary is None
            else args.fast_lambda_boundary_arbitrary
        )
        if boundary_origin_weight:
            active_losses.append("boundary_origin_wgf")
        if boundary_arbitrary_weight:
            active_losses.append("boundary_arbitrary_wgf")
        config["fast_active_losses"] = ",".join(active_losses)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in config.items():
            handle.write(f"{key}={value}\n")


def mnist_loader(args) -> DataLoader:
    return DataLoader(
        load_mnist_dataset(args.data_root, train=True, download=args.download),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device.type == "cuda"),
    )


def save_target_grid(data: torch.Tensor, out_dir):
    _save_mnist_grid(data.detach().cpu(), out_dir / "target_grid.png", "mnist target")


def sample_grid_shape(sample_count: int) -> tuple[int, int]:
    ncols = max(1, int(math.ceil(sample_count**0.5)))
    nrows = int(math.ceil(sample_count / ncols))
    return nrows, ncols


@torch.no_grad()
def make_mnist_mosaic(samples_flat: torch.Tensor, nrows: int, ncols: int) -> torch.Tensor:
    images = samples_flat.view(-1, 28, 28).detach().cpu().clamp(0.0, 1.0)
    mosaic = torch.zeros(nrows * 28, ncols * 28)
    for idx, image in enumerate(images[: nrows * ncols]):
        row = idx // ncols
        col = idx % ncols
        mosaic[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28] = image
    return mosaic


@torch.no_grad()
def save_flow_grid(samples_by_time: list[torch.Tensor], save_path, title: str):
    import matplotlib.pyplot as plt

    per_time_count = samples_by_time[0].shape[0]
    nrows, ncols = sample_grid_shape(per_time_count)
    panel_size = 2.6
    fig, axes = plt.subplots(1, len(PLOT_TIMES), figsize=(panel_size * len(PLOT_TIMES), panel_size + 0.5))
    axes = axes if isinstance(axes, (list, tuple)) else axes.reshape(-1)
    for ax, t, samples in zip(axes, PLOT_TIMES, samples_by_time):
        ax.imshow(make_mnist_mosaic(samples, nrows, ncols), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"t={float(t):.2f}")
        ax.axis("off")
    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.2, w_pad=0.2)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


@torch.no_grad()
def save_wgf_flow_map_grid(samples_by_steps: list[torch.Tensor], save_path, title: str):
    import matplotlib.pyplot as plt

    per_step_count = samples_by_steps[0].shape[0]
    nrows, ncols = sample_grid_shape(per_step_count)
    panel_size = 2.6
    fig, axes = plt.subplots(
        1,
        len(WGF_FLOW_MAP_SAMPLE_STEPS),
        figsize=(panel_size * len(WGF_FLOW_MAP_SAMPLE_STEPS), panel_size + 0.5),
    )
    axes = axes if isinstance(axes, (list, tuple)) else axes.reshape(-1)
    for ax, num_steps, samples in zip(axes, WGF_FLOW_MAP_SAMPLE_STEPS, samples_by_steps):
        ax.imshow(make_mnist_mosaic(samples, nrows, ncols), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{num_steps} steps")
        ax.axis("off")
    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.2, w_pad=0.2)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


@torch.no_grad()
def save_flow_training_samples(model, out_dir, args, step: int):
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    z = torch.randn(4, args.latent_dim, device=args.device)
    samples_by_time = [
        model(z, torch.full((4,), float(t), device=args.device))
        for t in PLOT_TIMES
    ]
    save_flow_grid(samples_by_time, sample_dir / f"generated_step_{step:06d}.png", f"step {step}")


@torch.no_grad()
def save_drift_training_samples(model, out_dir, args, step: int):
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    z = torch.randn(args.sample_count, args.latent_dim, device=args.device)
    samples = model(z)
    nrows, ncols = sample_grid_shape(args.sample_count)
    _save_mnist_grid(
        samples,
        sample_dir / f"generated_step_{step:06d}.png",
        f"step {step}",
        nrows=nrows,
        ncols=ncols,
    )


@torch.no_grad()
def save_ddim_samples(model, z: torch.Tensor, save_path, title: str, args):
    was_training = model.training
    model.eval()
    samples = ddim(
        z,
        partial(model_denoiser, model),
        args.ddim_steps,
        args.ddim_eta,
        **DDIM_SCHEDULE,
    )
    nrows, ncols = sample_grid_shape(z.shape[0])
    _save_mnist_grid(samples, save_path, title, nrows=nrows, ncols=ncols)
    model.train(was_training)


@torch.no_grad()
def save_wgf_flow_map_training_samples(model, z: torch.Tensor, out_dir, args, step: int):
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    was_training = model.training
    model.eval()
    samples_by_steps = [
        sample_wgf_flow_map(
            model,
            z,
            num_steps,
            args.sample_time_power,
        )
        for num_steps in WGF_FLOW_MAP_SAMPLE_STEPS
    ]
    save_wgf_flow_map_grid(samples_by_steps, sample_dir / f"generated_step_{step:06d}.png", f"step {step}")
    model.train(was_training)


def load_quality_classifier(device: torch.device):
    from eval_mnist_modes import SmallMnistClassifier

    checkpoint = REPO_ROOT / "runs" / "mnist_eval" / "mnist_classifier.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"quality classifier checkpoint not found: {checkpoint}")
    classifier = SmallMnistClassifier().to(device)
    payload = torch.load(checkpoint, map_location=device)
    classifier.load_state_dict(payload["model_state_dict"])
    classifier.eval()
    return classifier


@torch.no_grad()
def evaluate_wgf_flow_map_quality(model, classifier, z: torch.Tensor, args, step: int) -> dict:
    from eval_mnist_modes import inception_score

    was_training = model.training
    model.eval()
    row = {"step": step}
    for num_steps in WGF_FLOW_MAP_SAMPLE_STEPS:
        samples = []
        probabilities = []
        for z_batch in z.split(args.batch):
            generated = sample_wgf_flow_map(model, z_batch, num_steps, args.sample_time_power)
            samples.append(generated.cpu())
            logits = classifier(generated.view(-1, 1, 28, 28))
            probabilities.append(logits.softmax(dim=1).cpu())
        generated = torch.cat(samples)
        probs = torch.cat(probabilities)
        proportions = torch.bincount(probs.argmax(dim=1), minlength=10).float() / probs.shape[0]
        score, _ = inception_score(probs)
        prefix = f"nfe_{num_steps}"
        row[f"{prefix}_inception_score"] = score
        row[f"{prefix}_confidence"] = float(probs.max(dim=1).values.mean())
        row[f"{prefix}_mode_entropy"] = float(-(proportions * proportions.clamp_min(1e-8).log()).sum())
        row[f"{prefix}_active_modes"] = int((proportions >= 0.01).sum())
        row[f"{prefix}_sample_mean"] = float(generated.mean())
        row[f"{prefix}_sample_std"] = float(generated.std())
        for digit, proportion in enumerate(proportions):
            row[f"{prefix}_mode_{digit}"] = float(proportion)
    model.train(was_training)
    return row


@torch.no_grad()
def save_flow_diagnostics(model, out_dir, args):
    z = torch.randn(args.diag_batch, args.latent_dim, device=args.device)
    shown = evaluate_flow_map(model, z, PLOT_TIMES.to(args.device))
    save_flow_grid([shown[:, i] for i in range(len(PLOT_TIMES))], out_dir / "flow_grid.png", "flow")


@torch.no_grad()
def save_drift_diagnostics(model, out_dir, args):
    z = torch.randn(args.diag_batch, args.latent_dim, device=args.device)
    generated = model(z)
    _save_mnist_grid(generated, out_dir / "generated_grid.png", "generated")


@torch.no_grad()
def save_wgf_flow_map_diagnostics(model, out_dir, args):
    was_training = model.training
    model.eval()
    z = torch.randn(args.diag_batch, args.latent_dim, device=args.device)
    samples_by_steps = [
        sample_wgf_flow_map(
            model,
            z,
            num_steps,
            args.sample_time_power,
        )
        for num_steps in WGF_FLOW_MAP_SAMPLE_STEPS
    ]
    save_wgf_flow_map_grid(samples_by_steps, out_dir / "wgf_flow_map_grid.png", "wgf_flow_map")
    save_transition_map_diagnostics(model, z, out_dir)
    model.train(was_training)


@torch.no_grad()
def save_transition_map_diagnostics(model, z: torch.Tensor, out_dir):
    import matplotlib.pyplot as plt

    times = PLOT_TIMES.to(device=z.device, dtype=z.dtype)
    batch = z.shape[0]
    shown_count = min(16, batch)
    zeros = torch.zeros(batch, device=z.device, dtype=z.dtype)
    canonical = [
        model(z, zeros, torch.full_like(zeros, float(s)))
        for s in times
    ]
    nrows, ncols = sample_grid_shape(shown_count)
    fig, axes = plt.subplots(len(times), len(times), figsize=(13, 13))
    rows = []
    for i, t in enumerate(times):
        for j, s in enumerate(times):
            ax = axes[i, j]
            if j < i:
                ax.axis("off")
                continue
            mapped = canonical[i] if i == j else model(
                canonical[i],
                torch.full_like(zeros, float(t)),
                torch.full_like(zeros, float(s)),
            )
            semigroup_rmse = (mapped - canonical[j]).square().mean().sqrt()
            ax.imshow(
                make_mnist_mosaic(mapped[:shown_count], nrows, ncols),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(f"{float(t):.2f}->{float(s):.2f}\nRMSE {float(semigroup_rmse):.3f}")
            ax.axis("off")
            if i == j:
                continue
            mapped_delta = mapped - canonical[i]
            target_delta = canonical[j] - canonical[i]
            rows.append(
                {
                    "start_time": float(t),
                    "end_time": float(s),
                    "semigroup_rmse": float(semigroup_rmse),
                    "relative_rmse": float(
                        semigroup_rmse / target_delta.square().mean().sqrt().clamp_min(1e-8)
                    ),
                    "delta_cosine": float(
                        F.cosine_similarity(mapped_delta, target_delta, dim=1).mean()
                    ),
                    "target_progress": float(
                        (
                            (mapped_delta * target_delta).sum(dim=1)
                            / target_delta.square().sum(dim=1).clamp_min(1e-8)
                        ).mean()
                    ),
                }
            )
    fig.suptitle("Direct transitions from canonical states", y=0.995)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98), pad=0.2)
    fig.savefig(out_dir / "transition_map_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_history(out_dir / "transition_map_metrics.csv", rows)


def train_flow(args):
    model = TimeConditionedMnistConvGenerator(args.latent_dim, args.base_channels).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = run_dir("flow", args.run_label)
    write_config(out_dir / "config.txt", args, "flow")
    loader = mnist_loader(args)
    data_iter = iter(loader)
    history = []
    saved_target = False

    for _ in trange(args.steps, desc="flow:mnist"):
        data_iter, (data, _) = _next_batch(data_iter, loader)
        data = data.to(args.device, non_blocking=True).flatten(start_dim=1)
        if not saved_target:
            save_target_grid(data, out_dir)
            saved_target = True

        z = torch.randn(data.shape[0], args.latent_dim, device=args.device)
        z_ref = torch.randn(data.shape[0], args.latent_dim, device=args.device)

        loss, stats = euler_flow_matching_loss(
            model,
            z,
            z_ref,
            data,
            dt=args.dt,
            schedule=args.schedule,
            time_scale=args.time_scale,
            eps=args.eps,
            lambda_terminal=args.lambda_terminal if args.terminal_loss else 0.0,
            velocity_kind=args.velocity,
            bandwidth=args.bandwidth,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(to_float_dict(stats))
        if args.sample_every > 0 and len(history) % args.sample_every == 0:
            save_flow_training_samples(model, out_dir, args, len(history))

    write_config(out_dir / "config.txt", args, "flow")
    write_history(out_dir / "metrics.csv", history)
    plot_history(
        out_dir / "loss_curve.png",
        history,
        ["loss_dyn", "loss_terminal", "velocity_norm"],
        "flow: mnist",
    )
    save_flow_diagnostics(model, out_dir, args)
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"saved flow run: {out_dir}")


def train_drift(args):
    model = MnistConvGenerator(args.latent_dim, args.base_channels).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = run_dir("drift", args.run_label)
    write_config(out_dir / "config.txt", args, "drift")
    loader = mnist_loader(args)
    data_iter = iter(loader)
    history = []
    saved_target = False

    for _ in trange(args.steps, desc="drift:mnist"):
        data_iter, (data, _) = _next_batch(data_iter, loader)
        data = data.to(args.device, non_blocking=True).flatten(start_dim=1)
        if not saved_target:
            save_target_grid(data, out_dir)
            saved_target = True

        z = torch.randn(data.shape[0], args.latent_dim, device=args.device)
        z_ref = torch.randn(data.shape[0], args.latent_dim, device=args.device)

        loss, stats = optimizer_time_drifting_loss(
            model,
            z,
            z_ref,
            data,
            step_size=args.eta,
            velocity_kind=args.velocity,
            bandwidth=args.bandwidth,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(to_float_dict(stats))
        if args.sample_every > 0 and len(history) % args.sample_every == 0:
            save_drift_training_samples(model, out_dir, args, len(history))

    write_config(out_dir / "config.txt", args, "drift")
    write_history(out_dir / "metrics.csv", history)
    plot_history(out_dir / "loss_curve.png", history, ["loss", "velocity_norm"], "drift: mnist")
    save_drift_diagnostics(model, out_dir, args)
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"saved drift run: {out_dir}")


def train_ddim(args):
    model = TimeConditionedMnistDenoiser(args.latent_dim, args.base_channels).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = run_dir("ddim", args.run_label)
    write_config(out_dir / "config.txt", args, "ddim")
    loader = mnist_loader(args)
    data_iter = iter(loader)
    alphas, sigmas = mk_schedule(n_steps=args.ddim_steps, **DDIM_SCHEDULE)
    alphas, sigmas = alphas.to(args.device), sigmas.to(args.device)
    denoiser_fn = partial(model_denoiser, model)
    history = []
    saved_target = False
    fixed_z = torch.randn(args.sample_count, args.latent_dim, device=args.device)

    for _ in trange(args.steps, desc="ddim:mnist"):
        data_iter, (data, _) = _next_batch(data_iter, loader)
        data = data.to(args.device, non_blocking=True).flatten(start_dim=1)
        if not saved_target:
            save_target_grid(data, out_dir)
            saved_target = True

        t = torch.randint(args.ddim_steps, (data.shape[0],), device=args.device)
        x_t = q_sample(data, t, alphas, sigmas)
        loss = diffusion_loss(data, x_t, t, alphas, sigmas, denoiser_fn)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append({"loss": float(loss.detach().cpu())})
        if args.sample_every > 0 and len(history) % args.sample_every == 0:
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            save_ddim_samples(
                model,
                fixed_z,
                sample_dir / f"generated_step_{len(history):06d}.png",
                f"step {len(history)}",
                args,
            )

    write_config(out_dir / "config.txt", args, "ddim")
    write_history(out_dir / "metrics.csv", history)
    plot_history(out_dir / "loss_curve.png", history, ["loss"], "ddim: mnist")
    diagnostic_z = torch.randn(args.diag_batch, args.latent_dim, device=args.device)
    save_ddim_samples(model, diagnostic_z, out_dir / "generated_grid.png", "generated", args)
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"saved ddim run: {out_dir}")


def train_wgf_flow_map(args, *, fast: bool = False):
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if args.ema_warmup_steps < 0:
        raise ValueError("ema_warmup_steps must be non-negative")
    model = TransitionConditionedMnistConvGenerator(
        args.latent_dim,
        args.base_channels,
    ).to(args.device)
    ema_model = (
        AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
            use_buffers=True,
        )
        if fast and args.ema_decay > 0
        else None
    )
    ema_started = False
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    method = "fast_wgf_flow_map" if fast else "wgf_flow_map"
    out_dir = run_dir(method, args.run_label)
    write_config(out_dir / "config.txt", args, method)
    loader = mnist_loader(args)
    data_iter = iter(loader)
    history = []
    quality_history = []
    best_nfe4_score = float("-inf")
    saved_target = False
    fixed_generator = torch.Generator().manual_seed(args.seed + 1)
    fixed_z = torch.randn(args.sample_count, args.latent_dim, generator=fixed_generator).to(args.device)
    quality_z = torch.randn(args.quality_count, args.latent_dim, generator=fixed_generator).to(args.device)
    quality_classifier = load_quality_classifier(args.device) if args.quality_every > 0 else None
    data_permutation = None
    if args.training_data_control == "pixel_permutation":
        control_generator = torch.Generator().manual_seed(args.data_control_seed)
        data_permutation = torch.randperm(args.latent_dim, generator=control_generator).to(args.device)

    for step in trange(args.steps, desc=f"{method}:mnist"):
        data_iter, (data, _) = _next_batch(data_iter, loader)
        data = data.to(args.device, non_blocking=True).flatten(start_dim=1)
        if data_permutation is not None:
            data = data[:, data_permutation]
        if not saved_target:
            save_target_grid(data, out_dir)
            saved_target = True

        z = torch.randn(data.shape[0], args.latent_dim, device=args.device)
        z_ref = torch.randn(data.shape[0], args.latent_dim, device=args.device)
        progress = step / max(args.steps - 1, 1)
        min_span = args.fast_distill_min_span if fast else args.distill_min_span
        max_span = args.fast_distill_max_span if fast else 1.0
        distill_span = min_span + (max_span - min_span) * progress

        if fast:
            loss, stats = fast_wgf_flow_map_loss(
                model,
                z,
                z_ref,
                data,
                dt=args.dt,
                schedule=args.schedule,
                time_scale=args.wgf_flow_map_time_scale,
                eps=args.eps,
                path_weight=args.fast_path_weight,
                lambda_wgf=args.fast_lambda_wgf,
                lambda_lagrangian=args.fast_lambda_lagrangian,
                lambda_eulerian=args.fast_lambda_eulerian,
                lambda_boundary=args.fast_lambda_boundary,
                lambda_boundary_origin=args.fast_lambda_boundary_origin,
                lambda_boundary_arbitrary=args.fast_lambda_boundary_arbitrary,
                distill_max_span=distill_span,
                time_groups=args.time_groups,
                velocity_kind=args.velocity,
                bandwidth=args.bandwidth,
            )
        else:
            loss, stats = wgf_flow_map_loss(
                model,
                z,
                z_ref,
                data,
                eps_t=args.dt,
                schedule=args.schedule,
                time_scale=args.wgf_flow_map_time_scale,
                eps=args.eps,
                lambda_sg=args.lambda_sg,
                semigroup_composition_weight=args.psd_weight,
                lambda_terminal=args.lambda_terminal,
                terminal_steps=args.terminal_steps,
                terminal_time_power=args.sample_time_power,
                distill_max_span=distill_span,
                time_groups=args.time_groups,
                path_weight=args.path_weight,
                velocity_kind=args.velocity,
                bandwidth=args.bandwidth,
            )
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip if args.grad_clip > 0 else float("inf"),
        )
        opt.step()
        if ema_model is not None and step + 1 >= args.ema_warmup_steps:
            ema_model.update_parameters(model)
            ema_started = True
        eval_model = ema_model.module if ema_started else model
        stats["step"] = step + 1
        stats["grad_norm"] = grad_norm.detach()
        stats["learning_rate"] = opt.param_groups[0]["lr"]
        history.append(to_float_dict(stats))
        completed_steps = step + 1
        wrote_artifact = False
        if args.sample_every > 0 and completed_steps % args.sample_every == 0:
            save_wgf_flow_map_training_samples(eval_model, fixed_z, out_dir, args, completed_steps)
            wrote_artifact = True
        if args.quality_every > 0 and completed_steps % args.quality_every == 0:
            quality_row = evaluate_wgf_flow_map_quality(
                eval_model,
                quality_classifier,
                quality_z,
                args,
                completed_steps,
            )
            quality_history.append(quality_row)
            write_history(out_dir / "quality_metrics.csv", quality_history)
            if fast and quality_row["nfe_4_inception_score"] > best_nfe4_score:
                best_nfe4_score = quality_row["nfe_4_inception_score"]
                torch.save(eval_model.state_dict(), out_dir / "model_best_nfe4.pth")
            wrote_artifact = True
        if args.checkpoint_every > 0 and completed_steps % args.checkpoint_every == 0:
            checkpoint_dir = out_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(eval_model.state_dict(), checkpoint_dir / f"model_step_{completed_steps:06d}.pth")
            wrote_artifact = True
        if wrote_artifact:
            write_history(out_dir / "metrics.csv", history)

    write_config(out_dir / "config.txt", args, method)
    write_history(out_dir / "metrics.csv", history)
    plot_history(
        out_dir / "loss_curve.png",
        history,
        (
            [
                "loss_wgf",
                "loss_lagrangian",
                "loss_eulerian",
                "loss_boundary_origin",
                "loss_boundary_arbitrary",
            ]
            if fast
            else ["loss_wgf", "loss_sg", "loss_terminal"]
        ),
        f"{method}: mnist",
    )
    plot_history(
        out_dir / "diagnostics_curve.png",
        history,
        ["velocity_norm", "boundary_origin_velocity_norm", "grad_norm"],
        "wgf_flow_map diagnostics: mnist",
    )
    eval_model = ema_model.module if ema_started else model
    save_wgf_flow_map_diagnostics(eval_model, out_dir, args)
    torch.save(eval_model.state_dict(), out_dir / "model_final.pth")
    if ema_started:
        torch.save(model.state_dict(), out_dir / "model_raw_final.pth")
    print(f"saved {method} run: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["flow", "drift", "ddim", "wgf_flow_map", "fast_wgf_flow_map", "both"],
        default="wgf_flow_map",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=MNIST_DIM)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--wgf-flow-map-time-scale", type=float, default=100.0)
    parser.add_argument("--schedule", choices=["constant", "rational"], default="constant")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--diag-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-warmup-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--ddim-steps", type=int, default=1000)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--terminal-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda-sg", type=float, default=0.0)
    parser.add_argument("--fast-path-weight", type=float, default=0.1)
    parser.add_argument("--fast-lambda-wgf", type=float, default=1.0)
    parser.add_argument("--fast-lambda-lagrangian", type=float, default=1.0)
    parser.add_argument("--fast-lambda-eulerian", type=float, default=100.0)
    parser.add_argument("--fast-lambda-boundary", type=float, default=5.0)
    parser.add_argument("--fast-lambda-boundary-origin", type=float, default=None)
    parser.add_argument("--fast-lambda-boundary-arbitrary", type=float, default=None)
    parser.add_argument("--psd-weight", type=float, default=0.0)
    parser.add_argument("--lambda-terminal", type=float, default=0.0)
    parser.add_argument("--terminal-steps", type=int, default=4)
    parser.add_argument("--distill-min-span", type=float, default=1.0)
    parser.add_argument("--fast-distill-min-span", type=float, default=0.25)
    parser.add_argument("--fast-distill-max-span", type=float, default=0.5)
    parser.add_argument("--sample-time-power", type=float, default=1.0)
    parser.add_argument("--time-groups", type=int, default=4)
    parser.add_argument("--path-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--quality-every", type=int, default=0)
    parser.add_argument("--quality-count", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--velocity", choices=["mmd", "kl"], default="kl")
    parser.add_argument("--bandwidth", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--training-data-control",
        choices=["none", "pixel_permutation"],
        default="none",
    )
    parser.add_argument("--data-control-seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-count", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device(args.device)
    args.data_root = REPO_ROOT / "data"

    methods = ["flow", "drift"] if args.method == "both" else [args.method]
    for method in methods:
        if method == "flow":
            train_flow(args)
        elif method == "drift":
            train_drift(args)
        elif method == "ddim":
            train_ddim(args)
        elif method == "wgf_flow_map":
            train_wgf_flow_map(args)
        else:
            train_wgf_flow_map(args, fast=True)


if __name__ == "__main__":
    main()
