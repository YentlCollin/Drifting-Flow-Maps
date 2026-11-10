from __future__ import annotations

import argparse
import csv
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from local_paths import REPO_ROOT
from src.ddim import ddim, diffusion_loss, mk_schedule, model_denoiser, q_sample
from src.model import MLP, TimeConditionedMLP, TransitionConditionedMLP
from src.wgf import (
    evaluate_flow_map,
    euler_flow_matching_loss,
    fast_wgf_flow_map_loss,
    mmd2,
    optimizer_time_drifting_loss,
    sample_wgf_flow_map,
    wgf_flow_map_loss,
)


PLOT_TIMES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
WGF_FLOW_MAP_SAMPLE_STEPS = [1, 2, 4, 8, 16, 50]
DDIM_SCHEDULE = {"schedule_type": "vp", "beta_min": 1e-4, "beta_max": 2e-2}
DIM = 2


def sample_target(name: str, n: int, device, args) -> torch.Tensor:
    if name == "gaussian":
        return torch.randn(n, DIM, device=device) * 0.35 + 2.0

    if name == "gaussian_ring":
        angles = torch.linspace(0, 2 * torch.pi, args.ring_modes + 1, device=device)[:-1]
        means = args.ring_radius * torch.stack([angles.cos(), angles.sin()], dim=1)
        idx = torch.randint(args.ring_modes, (n,), device=device)
        return means[idx] + args.mode_std * torch.randn(n, DIM, device=device)

    weights = torch.tensor([0.7, 0.3], device=device)
    direction = torch.tensor([1.0, 1.0], device=device) / 2.0**0.5
    means = torch.stack([
        args.mixture_separation * 0.5 * direction,
        -args.mixture_separation * 0.5 * direction,
    ])
    idx = torch.multinomial(weights, n, replacement=True)
    return means[idx] + args.mode_std * torch.randn(n, DIM, device=device)


def run_dir(method: str, target: str):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = REPO_ROOT / "runs" / "toy_wgf" / method / target / stamp
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


def write_config(path, args, method: str, target: str):
    config = vars(args).copy()
    config["method"] = method
    config["target"] = target
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
        ):
            config.pop(key)
        active_losses = []
        if args.fast_path_weight:
            active_losses.append("wgf_origin_and_local")
            if args.fast_lambda_lagrangian:
                active_losses.append("lagrangian_fd")
            if args.fast_lambda_eulerian:
                active_losses.append("eulerian_fd")
        if args.fast_lambda_sg:
            active_losses.append("origin_semigroup")
        if args.fast_lambda_boundary:
            active_losses.extend(("boundary_origin_wgf", "boundary_arbitrary_wgf"))
        config["fast_active_losses"] = ",".join(active_losses)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in config.items():
            handle.write(f"{key}={value}\n")


def plot_history(path, history: list[dict], keys: list[str], title: str):
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


def plot_points(ax, x: torch.Tensor, title: str, color: str):
    x = x.detach().cpu()
    ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.25, color=color)
    ax.set_title(title)
    ax.set_aspect("equal")


def plot_overlay(ax, model_samples: torch.Tensor, target_samples: torch.Tensor, title: str):
    target_samples = target_samples.detach().cpu()
    model_samples = model_samples.detach().cpu()
    ax.scatter(target_samples[:, 0], target_samples[:, 1], s=4, alpha=0.18, color="tab:blue", label="target")
    ax.scatter(model_samples[:, 0], model_samples[:, 1], s=4, alpha=0.28, color="tab:orange", label="model")
    ax.set_title(title)
    ax.set_aspect("equal")


@torch.no_grad()
def save_flow_diagnostics(model, target: str, out_dir, args):
    z = torch.randn(args.diag_batch, DIM, device=args.device)
    data = sample_target(target, args.diag_batch, args.device, args)
    shown = evaluate_flow_map(model, z, PLOT_TIMES.to(args.device))

    fig, axes = plt.subplots(1, len(PLOT_TIMES) + 1, figsize=(18.5, 3.2))
    plot_points(axes[0], z, "prior", "tab:gray")
    for i, t in enumerate(PLOT_TIMES):
        plot_overlay(axes[i + 1], shown[:, i], data, f"t={float(t):.2f}")
    axes[-1].legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "flow_samples.png", dpi=150)
    plt.close(fig)

    times = torch.linspace(0, 1, 21, device=args.device)
    path = evaluate_flow_map(model, z, times)
    mmd = [
        float(mmd2(path[:, i], data, args.bandwidth).clamp_min(0).sqrt().cpu())
        for i in range(len(times))
    ]
    init_mse = torch.mean((path[:, 0] - z).square()).item()
    smooth_step = (path[:, 1:] - path[:, :-1]).norm(dim=-1).mean().item()

    plt.figure(figsize=(6, 4))
    plt.plot(times.cpu(), mmd, marker="o")
    plt.xlabel("t")
    plt.ylabel("MMD(q_t, p)")
    plt.title("MMD along learned flow")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "mmd_vs_time.png", dpi=150)
    plt.close()

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"init_mse={init_mse:.8f}\n")
        handle.write(f"terminal_mmd={mmd[-1]:.8f}\n")
        handle.write(f"mean_step_distance={smooth_step:.8f}\n")


@torch.no_grad()
def save_drift_diagnostics(model, target: str, out_dir, args):
    z = torch.randn(args.diag_batch, DIM, device=args.device)
    generated = model(z)
    data = sample_target(target, args.diag_batch, args.device, args)
    final_mmd = float(mmd2(generated, data, args.bandwidth).clamp_min(0).sqrt().cpu())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    plot_points(axes[0], z, "prior", "tab:gray")
    plot_overlay(axes[1], generated, data, "generated vs target")
    axes[1].legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "drift_samples.png", dpi=150)
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"final_mmd={final_mmd:.8f}\n")


@torch.no_grad()
def save_ddim_diagnostics(model, target: str, out_dir, args):
    z = torch.randn(args.diag_batch, DIM, device=args.device)
    generated = ddim(
        z,
        partial(model_denoiser, model),
        args.ddim_steps,
        args.ddim_eta,
        **DDIM_SCHEDULE,
    )
    data = sample_target(target, args.diag_batch, args.device, args)
    final_mmd = float(mmd2(generated, data, args.bandwidth).clamp_min(0).sqrt().cpu())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    plot_points(axes[0], z, "prior", "tab:gray")
    plot_overlay(axes[1], generated, data, "generated vs target")
    axes[1].legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "ddim_samples.png", dpi=150)
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"final_mmd={final_mmd:.8f}\n")


@torch.no_grad()
def save_wgf_flow_map_diagnostics(model, target: str, out_dir, args):
    z = torch.randn(args.diag_batch, DIM, device=args.device)
    data = sample_target(target, args.diag_batch, args.device, args)

    fig, axes = plt.subplots(1, len(WGF_FLOW_MAP_SAMPLE_STEPS) + 1, figsize=(20.5, 3.2))
    plot_points(axes[0], z, "prior", "tab:gray")
    final_mmd = {}
    bridge_mass = {}
    for ax, num_steps in zip(axes[1:], WGF_FLOW_MAP_SAMPLE_STEPS):
        samples = sample_wgf_flow_map(
            model,
            z,
            num_steps,
            args.sample_time_power,
        )
        plot_overlay(ax, samples, data, f"{num_steps} steps")
        final_mmd[num_steps] = float(mmd2(samples, data, args.bandwidth).clamp_min(0).sqrt().cpu())
        if target == "gaussian_mixture":
            direction = torch.tensor([1.0, 1.0], device=args.device) / 2.0**0.5
            projection = samples @ direction
            bridge = projection.abs() < 0.25 * args.mixture_separation
            bridge_mass[num_steps] = float(bridge.float().mean().cpu())
    axes[-1].legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "wgf_flow_map_samples.png", dpi=150)
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        for num_steps in WGF_FLOW_MAP_SAMPLE_STEPS:
            handle.write(f"terminal_mmd_{num_steps}_steps={final_mmd[num_steps]:.8f}\n")
        for num_steps in WGF_FLOW_MAP_SAMPLE_STEPS:
            if num_steps in bridge_mass:
                handle.write(f"bridge_mass_{num_steps}_steps={bridge_mass[num_steps]:.8f}\n")


def train_flow(target: str, args):
    model = TimeConditionedMLP(DIM, args.hidden, DIM).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = run_dir("flow", target)
    write_config(out_dir / "config.txt", args, "flow", target)
    history = []

    for _ in trange(args.steps, desc=f"flow:{target}"):
        z = torch.randn(args.batch, DIM, device=args.device)
        z_ref = torch.randn(args.batch, DIM, device=args.device)
        data = sample_target(target, args.batch, args.device, args)

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

    write_history(out_dir / "metrics.csv", history)
    plot_history(
        out_dir / "loss_curve.png",
        history,
        ["loss_dyn", "loss_terminal", "velocity_norm"],
        f"flow: {target}",
    )
    save_flow_diagnostics(model, target, out_dir, args)
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"saved flow run: {out_dir}")


def train_drift(target: str, args):
    model = MLP(DIM, args.hidden, DIM).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = run_dir("drift", target)
    write_config(out_dir / "config.txt", args, "drift", target)
    history = []

    for _ in trange(args.steps, desc=f"drift:{target}"):
        z = torch.randn(args.batch, DIM, device=args.device)
        z_ref = torch.randn(args.batch, DIM, device=args.device)
        data = sample_target(target, args.batch, args.device, args)

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

    write_history(out_dir / "metrics.csv", history)
    plot_history(out_dir / "loss_curve.png", history, ["loss", "velocity_norm"], f"drift: {target}")
    save_drift_diagnostics(model, target, out_dir, args)
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"saved drift run: {out_dir}")


def train_ddim(target: str, args):
    model = TimeConditionedMLP(DIM, args.hidden, DIM).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = run_dir("ddim", target)
    write_config(out_dir / "config.txt", args, "ddim", target)
    alphas, sigmas = mk_schedule(n_steps=args.ddim_steps, **DDIM_SCHEDULE)
    alphas, sigmas = alphas.to(args.device), sigmas.to(args.device)
    denoiser_fn = partial(model_denoiser, model)
    history = []

    for _ in trange(args.steps, desc=f"ddim:{target}"):
        data = sample_target(target, args.batch, args.device, args)
        t = torch.randint(args.ddim_steps, (args.batch,), device=args.device)
        x_t = q_sample(data, t, alphas, sigmas)
        loss = diffusion_loss(data, x_t, t, alphas, sigmas, denoiser_fn)

        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append({"loss": float(loss.detach().cpu())})

    write_history(out_dir / "metrics.csv", history)
    plot_history(out_dir / "loss_curve.png", history, ["loss"], f"ddim: {target}")
    model.eval()
    save_ddim_diagnostics(model, target, out_dir, args)
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"saved ddim run: {out_dir}")


def train_wgf_flow_map(target: str, args, *, fast: bool = False):
    model = TransitionConditionedMLP(DIM, args.hidden, DIM).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    method = "fast_wgf_flow_map" if fast else "wgf_flow_map"
    out_dir = run_dir(method, target)
    write_config(out_dir / "config.txt", args, method, target)
    history = []

    for step in trange(args.steps, desc=f"{method}:{target}"):
        z = torch.randn(args.batch, DIM, device=args.device)
        z_ref = torch.randn(args.batch, DIM, device=args.device)
        data = sample_target(target, args.batch, args.device, args)
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
                lambda_lagrangian=args.fast_lambda_lagrangian,
                lambda_eulerian=args.fast_lambda_eulerian,
                lambda_sg=args.fast_lambda_sg,
                lambda_boundary=args.fast_lambda_boundary,
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
                dt=args.dt,
                schedule=args.schedule,
                time_scale=args.wgf_flow_map_time_scale,
                eps=args.eps,
                lambda_sg=args.lambda_sg,
                semigroup_composition_weight=args.psd_weight,
                lambda_terminal=args.lambda_terminal,
                terminal_steps=args.terminal_steps,
                terminal_time_power=args.sample_time_power,
                distill_max_span=distill_span,
                velocity_kind=args.velocity,
                bandwidth=args.bandwidth,
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(to_float_dict(stats))

    write_history(out_dir / "metrics.csv", history)
    plot_history(
        out_dir / "loss_curve.png",
        history,
        (
            [
                "loss_wgf",
                "loss_lagrangian",
                "loss_eulerian",
                *(["loss_sg"] if args.fast_lambda_sg else []),
                "loss_boundary_origin",
                "loss_boundary_arbitrary",
            ]
            if fast
            else ["loss_wgf", "loss_sg", "loss_terminal"]
        ),
        f"{method}: {target}",
    )
    plot_history(
        out_dir / "diagnostics_curve.png",
        history,
        (
            ["velocity_norm", "origin_residual_ratio", "local_residual_ratio"]
            if fast
            else ["velocity_norm", "semigroup_error"]
        ),
        f"{method} diagnostics: {target}",
    )
    save_wgf_flow_map_diagnostics(model, target, out_dir, args)
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"saved {method} run: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["flow", "drift", "ddim", "wgf_flow_map", "fast_wgf_flow_map", "both"],
        default="fast_wgf_flow_map",
    )
    parser.add_argument("--target", choices=["gaussian", "gaussian_mixture", "gaussian_ring", "both"], default="both")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--wgf-flow-map-time-scale", type=float, default=50.0)
    parser.add_argument("--schedule", choices=["constant", "rational"], default="constant")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--diag-batch", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--eta", type=float, default=0.35)
    parser.add_argument("--ddim-steps", type=int, default=1000)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--terminal-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda-sg", type=float, default=0.5)
    parser.add_argument("--fast-path-weight", type=float, default=0.1)
    parser.add_argument("--fast-lambda-lagrangian", type=float, default=1.0)
    parser.add_argument("--fast-lambda-eulerian", type=float, default=100.0)
    parser.add_argument("--fast-lambda-sg", type=float, default=1.0)
    parser.add_argument("--fast-lambda-boundary", type=float, default=5.0)
    parser.add_argument("--psd-weight", type=float, default=0.0)
    parser.add_argument("--lambda-terminal", type=float, default=0.1)
    parser.add_argument("--terminal-steps", type=int, default=8)
    parser.add_argument("--distill-min-span", type=float, default=1.0)
    parser.add_argument("--fast-distill-min-span", type=float, default=0.25)
    parser.add_argument("--fast-distill-max-span", type=float, default=0.5)
    parser.add_argument("--time-groups", type=int, default=4)
    parser.add_argument("--sample-time-power", type=float, default=1.0)
    parser.add_argument("--mixture-separation", type=float, default=10)
    parser.add_argument("--mode-std", type=float, default=0.25)
    parser.add_argument("--ring-modes", type=int, default=8)
    parser.add_argument("--ring-radius", type=float, default=3.5)
    parser.add_argument("--velocity", choices=["mmd", "kl"], default="kl")
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device(args.device)

    targets = ["gaussian", "gaussian_mixture", "gaussian_ring"] if args.target == "both" else [args.target]
    methods = ["flow", "drift"] if args.method == "both" else [args.method]

    for target in targets:
        for method in methods:
            if method == "flow":
                train_flow(target, args)
            elif method == "drift":
                train_drift(target, args)
            elif method == "ddim":
                train_ddim(target, args)
            elif method == "fast_wgf_flow_map":
                train_wgf_flow_map(target, args, fast=True)
            else:
                train_wgf_flow_map(target, args)


if __name__ == "__main__":
    main()
