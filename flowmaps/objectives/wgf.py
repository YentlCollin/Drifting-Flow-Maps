from __future__ import annotations

import torch
import torch.nn.functional as F

RBF_GEOMETRIES = ("raw", "dimension_normalized")


def _canonical_rbf_geometry(rbf_geometry: str) -> str:
    aliases = {
        "raw": "raw",
        "euclidean": "raw",
        "dimension_normalized": "dimension_normalized",
        "dimension-normalized": "dimension_normalized",
        "normalized": "dimension_normalized",
    }
    try:
        return aliases[rbf_geometry]
    except KeyError as error:
        choices = ", ".join(RBF_GEOMETRIES)
        raise ValueError(
            f"unknown RBF geometry: {rbf_geometry!r}; expected one of {choices}"
        ) from error


def _feature_dimension(x: torch.Tensor) -> int:
    if x.ndim < 2 or x.shape[-1] < 1:
        raise ValueError("RBF inputs must have a non-empty feature dimension")
    return x.shape[-1]


def _distance_divisor(x: torch.Tensor, rbf_geometry: str) -> float:
    geometry = _canonical_rbf_geometry(rbf_geometry)
    return float(_feature_dimension(x)) if geometry == "dimension_normalized" else 1.0


def time_batch(t, batch_size: int, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(t, device=like.device, dtype=like.dtype).expand(batch_size)


def median_bandwidth(
    *batches: torch.Tensor,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    points = torch.cat([x.detach() for x in batches], dim=0)
    distances = torch.pdist(points)
    if _canonical_rbf_geometry(rbf_geometry) == "dimension_normalized":
        distances = distances / _feature_dimension(points) ** 0.5
    distances = distances[distances > 0]
    if distances.numel() == 0:
        return torch.tensor(1.0, device=points.device, dtype=points.dtype)
    return distances.median().clamp_min(1e-6)


def _bandwidth_tensor(
    bandwidth,
    *batches: torch.Tensor,
    rbf_geometry: str,
) -> torch.Tensor:
    x = batches[0]
    return (
        median_bandwidth(*batches, rbf_geometry=rbf_geometry)
        if bandwidth is None
        else torch.as_tensor(
            bandwidth,
            device=x.device,
            dtype=x.dtype,
        )
    )


def _rbf_logits(
    diff: torch.Tensor,
    bandwidth: torch.Tensor,
    rbf_geometry: str,
) -> torch.Tensor:
    # In dimension-normalized geometry the bandwidth is measured per coordinate:
    # ||x-y||^2 becomes its coordinate mean.  Velocity routines deliberately do
    # not divide their gradient by D below; that is the requested D-fold physical
    # time/energy compensation which preserves per-coordinate drift magnitude.
    dist2 = diff.square().sum(dim=-1) / _distance_divisor(diff, rbf_geometry)
    return -dist2 / (2.0 * bandwidth.square())


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth=None,
    *,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    h = _bandwidth_tensor(bandwidth, x, y, rbf_geometry=rbf_geometry)
    dist2 = torch.cdist(x, y).square() / _distance_divisor(x, rbf_geometry)
    return torch.exp(-dist2 / (2.0 * h.square()))


def mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth=None,
    *,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    h = _bandwidth_tensor(bandwidth, x, y, rbf_geometry=rbf_geometry)
    return (
        gaussian_kernel(x, x, h, rbf_geometry=rbf_geometry).mean()
        + gaussian_kernel(y, y, h, rbf_geometry=rbf_geometry).mean()
        - 2.0 * gaussian_kernel(x, y, h, rbf_geometry=rbf_geometry).mean()
    )


def _empty_support_diagnostics(x: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    return {
        "norm": zero,
        "effective_sample_size": zero,
        "kernel_mean": zero,
        "kernel_nonzero_fraction": zero,
    }


def _rbf_support_term(
    x: torch.Tensor,
    support: torch.Tensor,
    bandwidth: torch.Tensor,
    *,
    rbf_geometry: str,
    normalize_weights: bool,
    same_batch: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if support.shape[-1] != x.shape[-1]:
        raise ValueError("x and support must have the same feature dimension")
    if support.shape[0] == 0:
        raise ValueError("RBF support must contain at least one sample")
    if same_batch and x.shape[0] != support.shape[0]:
        raise ValueError(
            "same_batch=True requires x and support to have the same batch size"
        )
    if same_batch and x.shape[0] <= 1:
        return torch.zeros_like(x), _empty_support_diagnostics(x)

    diff = support.unsqueeze(0) - x.unsqueeze(1)
    logits = _rbf_logits(diff, bandwidth, rbf_geometry)
    valid_count = support.shape[0]
    if same_batch:
        logits = logits.clone()
        logits.fill_diagonal_(-torch.inf)
        valid_count -= 1

    raw_kernel = logits.exp()
    if normalize_weights:
        weights = logits.softmax(dim=1)
        weighted_diff = (weights.unsqueeze(-1) * diff).sum(dim=1)
    else:
        weights = raw_kernel
        weighted_diff = (weights.unsqueeze(-1) * diff).sum(dim=1) / valid_count

    # The denominator intentionally stays h^2 for both geometries.  Differentiating
    # mean squared coordinate distance would introduce 1/D; omitting it scales WGF
    # time by D and keeps a replicated coordinate's velocity unchanged as D grows.
    term = weighted_diff / bandwidth.square()
    weight_mass = weights.sum(dim=1)
    tiny = torch.finfo(weights.dtype).tiny
    safe_mass = torch.where(weight_mass > 0, weight_mass, torch.ones_like(weight_mass))
    normalized_for_ess = weights / safe_mass.unsqueeze(1)
    normalized_square_mass = normalized_for_ess.square().sum(dim=1)
    ess = torch.where(
        weight_mass > 0,
        normalized_square_mass.clamp_min(tiny).reciprocal(),
        torch.zeros_like(weight_mass),
    )
    diagnostics = {
        "norm": term.norm(dim=1).mean(),
        "effective_sample_size": ess.mean(),
        "kernel_mean": (raw_kernel.sum(dim=1) / valid_count).mean(),
        "kernel_nonzero_fraction": (raw_kernel > 0)
        .to(x.dtype)
        .sum(dim=1)
        .div(valid_count)
        .mean(),
    }
    return term, diagnostics


def _prefix_support_diagnostics(
    attraction: dict[str, torch.Tensor],
    repulsion: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    diagnostics = {
        **{f"attraction_{key}": value for key, value in attraction.items()},
        **{f"repulsion_{key}": value for key, value in repulsion.items()},
    }
    diagnostics["attraction_ess"] = diagnostics["attraction_effective_sample_size"]
    diagnostics["repulsion_ess"] = diagnostics["repulsion_effective_sample_size"]
    return diagnostics


def _mmd_wgf_velocity_and_diagnostics(
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool,
    rbf_geometry: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    h = _bandwidth_tensor(bandwidth, x, data, reference, rbf_geometry=rbf_geometry)
    data_term, attraction = _rbf_support_term(
        x,
        data,
        h,
        rbf_geometry=rbf_geometry,
        normalize_weights=False,
        same_batch=False,
    )
    ref_term, repulsion = _rbf_support_term(
        x,
        reference,
        h,
        rbf_geometry=rbf_geometry,
        normalize_weights=False,
        same_batch=same_batch,
    )
    return data_term - ref_term, _prefix_support_diagnostics(attraction, repulsion)


def mmd_wgf_velocity(
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool = False,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    velocity, _ = _mmd_wgf_velocity_and_diagnostics(
        x,
        data,
        reference,
        bandwidth,
        same_batch=same_batch,
        rbf_geometry=rbf_geometry,
    )
    return velocity


def _kde_score_and_diagnostics(
    x: torch.Tensor,
    support: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool,
    rbf_geometry: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    h = _bandwidth_tensor(bandwidth, x, support, rbf_geometry=rbf_geometry)
    return _rbf_support_term(
        x,
        support,
        h,
        rbf_geometry=rbf_geometry,
        normalize_weights=True,
        same_batch=same_batch,
    )


def kde_score(
    x: torch.Tensor,
    support: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool = False,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    score, _ = _kde_score_and_diagnostics(
        x,
        support,
        bandwidth,
        same_batch=same_batch,
        rbf_geometry=rbf_geometry,
    )
    return score


def _kl_kde_wgf_velocity_and_diagnostics(
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool,
    rbf_geometry: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Preserve the historical KL estimator: when bandwidth is automatic, each
    # KDE chooses its own median bandwidth rather than sharing a joint one.
    data_h = _bandwidth_tensor(bandwidth, x, data, rbf_geometry=rbf_geometry)
    reference_h = _bandwidth_tensor(bandwidth, x, reference, rbf_geometry=rbf_geometry)
    data_term, attraction = _rbf_support_term(
        x,
        data,
        data_h,
        rbf_geometry=rbf_geometry,
        normalize_weights=True,
        same_batch=False,
    )
    ref_term, repulsion = _rbf_support_term(
        x,
        reference,
        reference_h,
        rbf_geometry=rbf_geometry,
        normalize_weights=True,
        same_batch=same_batch,
    )
    return data_term - ref_term, _prefix_support_diagnostics(attraction, repulsion)


def kl_kde_wgf_velocity(
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool = False,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    velocity, _ = _kl_kde_wgf_velocity_and_diagnostics(
        x,
        data,
        reference,
        bandwidth,
        same_batch=same_batch,
        rbf_geometry=rbf_geometry,
    )
    return velocity


def _wgf_velocity_and_diagnostics(
    kind: str,
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool,
    rbf_geometry: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if kind == "mmd":
        return _mmd_wgf_velocity_and_diagnostics(
            x,
            data,
            reference,
            bandwidth,
            same_batch=same_batch,
            rbf_geometry=rbf_geometry,
        )
    if kind == "kl":
        return _kl_kde_wgf_velocity_and_diagnostics(
            x,
            data,
            reference,
            bandwidth,
            same_batch=same_batch,
            rbf_geometry=rbf_geometry,
        )
    raise ValueError(f"unknown velocity: {kind}")


def wgf_velocity(
    kind: str,
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool = False,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    velocity, _ = _wgf_velocity_and_diagnostics(
        kind,
        x,
        data,
        reference,
        bandwidth,
        same_batch=same_batch,
        rbf_geometry=rbf_geometry,
    )
    return velocity


@torch.no_grad()
def wgf_velocity_diagnostics(
    kind: str,
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth=None,
    *,
    same_batch: bool = False,
    rbf_geometry: str = "raw",
) -> dict[str, torch.Tensor]:
    """Return kernel health and attraction/repulsion diagnostics for a WGF field."""
    velocity, diagnostics = _wgf_velocity_and_diagnostics(
        kind,
        x,
        data,
        reference,
        bandwidth,
        same_batch=same_batch,
        rbf_geometry=rbf_geometry,
    )
    return {
        **{key: value.detach() for key, value in diagnostics.items()},
        "velocity_norm": velocity.norm(dim=1).mean().detach(),
        "velocity_rms": velocity.square().mean().sqrt().detach(),
    }


def distributed_is_active() -> bool:
    distributed = getattr(torch, "distributed", None)
    return bool(
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    )


def assemble_global_support(
    rank_supports: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    time_groups: int = 1,
) -> torch.Tensor:
    """Concatenate rank-local supports while preserving contiguous time groups.

    DDP gathers tensors rank-first.  A grouped KDE instead needs all ranks from
    time group 0, then all ranks from time group 1, etc., so that ``chunk`` keeps
    each global reference support at a single synchronized time.
    """
    if not rank_supports:
        raise ValueError("rank_supports must contain at least one tensor")
    if time_groups < 1:
        raise ValueError("time_groups must be positive")
    if any(support.shape[0] % time_groups for support in rank_supports):
        raise ValueError("time_groups must divide every rank-local support size")
    if time_groups == 1:
        return torch.cat(tuple(rank_supports), dim=0)
    chunks_by_rank = [support.chunk(time_groups, dim=0) for support in rank_supports]
    return torch.cat(
        [
            torch.cat([rank_chunks[group] for rank_chunks in chunks_by_rank], dim=0)
            for group in range(time_groups)
        ],
        dim=0,
    )


def gather_global_support(
    support: torch.Tensor,
    *,
    time_groups: int = 1,
) -> torch.Tensor:
    """All-gather a KDE support, or return it unchanged outside initialized DDP."""
    if not distributed_is_active() or torch.distributed.get_world_size() <= 1:
        return support
    if support.ndim < 1:
        raise ValueError("support must have a batch dimension")

    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor(
        [support.shape[0]], device=support.device, dtype=torch.int64
    )
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)
    if support.shape[0] < max_size:
        padding = support.new_zeros((max_size - support.shape[0], *support.shape[1:]))
        padded_support = torch.cat((support, padding), dim=0)
    else:
        padded_support = support
    gathered = [torch.empty_like(padded_support) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded_support.contiguous())
    rank_supports = [rank_support[:size] for rank_support, size in zip(gathered, sizes)]
    return assemble_global_support(rank_supports, time_groups=time_groups)


def synchronize_distributed_tensor(
    value: torch.Tensor, *, enabled: bool
) -> torch.Tensor:
    """Broadcast rank-zero values when requested; otherwise behave as identity."""
    if (
        not enabled
        or not distributed_is_active()
        or torch.distributed.get_world_size() <= 1
    ):
        return value
    synchronized = value.clone()
    torch.distributed.broadcast(synchronized, src=0)
    return synchronized


def sample_grouped_times(
    batch_size: int,
    time_groups: int,
    like: torch.Tensor,
    *,
    synchronize: bool = False,
) -> torch.Tensor:
    if time_groups < 1 or batch_size % time_groups:
        raise ValueError("time_groups must be positive and divide the batch size")
    shape = () if time_groups == 1 else (time_groups,)
    times = torch.rand(shape, device=like.device, dtype=like.dtype)
    times = synchronize_distributed_tensor(times, enabled=synchronize)
    if time_groups == 1:
        return times.expand(batch_size)
    return times.repeat_interleave(batch_size // time_groups)


def _configured_wgf_velocity(
    kind: str,
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth,
    *,
    rbf_geometry: str,
) -> torch.Tensor:
    # Keep the historical five-positional-argument call in raw mode.  Besides
    # preserving API behavior, this keeps lightweight research monkeypatches valid.
    if _canonical_rbf_geometry(rbf_geometry) == "raw":
        return wgf_velocity(kind, x, data, reference, bandwidth)
    return wgf_velocity(
        kind,
        x,
        data,
        reference,
        bandwidth,
        rbf_geometry=rbf_geometry,
    )


def grouped_wgf_velocity(
    kind: str,
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth,
    time_groups: int,
    *,
    rbf_geometry: str = "raw",
) -> torch.Tensor:
    if time_groups == 1:
        return _configured_wgf_velocity(
            kind,
            x,
            data,
            reference,
            bandwidth,
            rbf_geometry=rbf_geometry,
        )
    return torch.cat(
        [
            _configured_wgf_velocity(
                kind,
                x_group,
                data,
                reference_group,
                bandwidth,
                rbf_geometry=rbf_geometry,
            )
            for x_group, reference_group in zip(
                x.chunk(time_groups), reference.chunk(time_groups)
            )
        ]
    )


@torch.no_grad()
def grouped_wgf_velocity_diagnostics(
    kind: str,
    x: torch.Tensor,
    data: torch.Tensor,
    reference: torch.Tensor,
    bandwidth,
    time_groups: int,
    *,
    rbf_geometry: str = "raw",
    max_queries_per_group: int | None = None,
) -> dict[str, torch.Tensor]:
    x_groups = x.chunk(time_groups)
    reference_groups = reference.chunk(time_groups)
    group_diagnostics = []
    for x_group, reference_group in zip(x_groups, reference_groups):
        if max_queries_per_group is not None:
            x_group = x_group[:max_queries_per_group]
        group_diagnostics.append(
            wgf_velocity_diagnostics(
                kind,
                x_group,
                data,
                reference_group,
                bandwidth,
                rbf_geometry=rbf_geometry,
            )
        )
    return {
        key: torch.stack([diagnostics[key] for diagnostics in group_diagnostics]).mean()
        for key in group_diagnostics[0]
    }


def multiscale_mmd_wgf_velocity(
    x,
    data,
    reference,
    bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0),
    *,
    same_batch: bool = False,
    rbf_geometry: str = "raw",
):
    velocity = 0.0
    for h in bandwidths:
        velocity = velocity + mmd_wgf_velocity(
            x,
            data,
            reference,
            bandwidth=h,
            same_batch=same_batch,
            rbf_geometry=rbf_geometry,
        )
    return velocity / len(bandwidths)


def wgf_time(
    t, *, schedule: str = "constant", time_scale: float = 1.0, eps: float = 1e-3
):
    t = torch.as_tensor(t)
    if schedule == "constant":
        return time_scale * t
    if schedule == "rational":
        return time_scale * t / (1.0 - t + eps)
    raise ValueError(f"unknown beta schedule: {schedule}")


def beta_value(
    t, *, schedule: str = "constant", time_scale: float = 1.0, eps: float = 1e-3
):
    t = torch.as_tensor(t)
    if schedule == "constant":
        return torch.as_tensor(time_scale, device=t.device, dtype=t.dtype)
    if schedule == "rational":
        return time_scale * (1.0 + eps) / (1.0 - t + eps).square()
    raise ValueError(f"unknown beta schedule: {schedule}")


def schedule_delta(
    t,
    dt: float,
    *,
    schedule: str = "constant",
    time_scale: float = 1.0,
    eps: float = 1e-3,
):
    return transition_delta(
        t, t + dt, schedule=schedule, time_scale=time_scale, eps=eps
    )


def transition_delta(
    t, s, *, schedule: str = "constant", time_scale: float = 1.0, eps: float = 1e-3
):
    return (
        wgf_time(s, schedule=schedule, time_scale=time_scale, eps=eps)
        - wgf_time(t, schedule=schedule, time_scale=time_scale, eps=eps)
    ).clamp_min(1e-8)


def optimizer_time_drifting_loss(
    model,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    data: torch.Tensor,
    *,
    step_size: float,
    velocity_kind: str = "mmd",
    bandwidth=None,
    rbf_geometry: str = "raw",
    global_kde_support: bool = False,
):
    x = model(z)
    with torch.no_grad():
        ref = model(z_ref)
        data_support = gather_global_support(data) if global_kde_support else data
        ref_support = gather_global_support(ref) if global_kde_support else ref
        drift = _configured_wgf_velocity(
            velocity_kind,
            x,
            data_support,
            ref_support,
            bandwidth,
            rbf_geometry=rbf_geometry,
        )
        target = x + step_size * drift
    loss = F.mse_loss(x, target)
    return loss, {
        "loss": loss.detach(),
        "velocity_norm": drift.norm(dim=1).mean(),
    }


def euler_flow_matching_loss(
    model,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    data: torch.Tensor,
    *,
    dt: float,
    schedule: str = "constant",
    time_scale: float = 1.0,
    eps: float = 1e-3,
    lambda_terminal: float = 0.1,
    velocity_kind: str = "mmd",
    bandwidth=None,
    rbf_geometry: str = "raw",
):
    batch_size = z.shape[0]
    t = torch.rand((), device=z.device, dtype=z.dtype) * (1.0 - dt)
    delta_s = schedule_delta(
        t, dt, schedule=schedule, time_scale=time_scale, eps=eps
    ).reshape(1, 1)

    with torch.no_grad():
        x_t = model(z, time_batch(t, batch_size, z))
        ref_t = model(z_ref, time_batch(t, z_ref.shape[0], z_ref))
        drift = _configured_wgf_velocity(
            velocity_kind,
            x_t,
            data,
            ref_t,
            bandwidth,
            rbf_geometry=rbf_geometry,
        )
        # velocity = multiscale_mmd_wgf_velocity(x_t, data, ref_t)

    x_next = model(z, time_batch(t + dt, batch_size, z))
    loss_dyn = F.mse_loss((x_next - x_t) / delta_s, drift)

    loss_terminal = torch.zeros((), device=z.device, dtype=z.dtype)
    if lambda_terminal:
        x1 = model(z, torch.ones(batch_size, device=z.device, dtype=z.dtype))
        with torch.no_grad():
            ref1 = model(
                z_ref, torch.ones(z_ref.shape[0], device=z.device, dtype=z.dtype)
            )
            velocity1 = _configured_wgf_velocity(
                velocity_kind,
                x1,
                data,
                ref1,
                bandwidth,
                rbf_geometry=rbf_geometry,
            )
            # velocity1 = multiscale_mmd_wgf_velocity(x1, data, ref1)
            target1 = x1 + dt * velocity1
        loss_terminal = F.mse_loss(x1, target1)

    total = loss_dyn + lambda_terminal * loss_terminal

    return total, {
        "loss": total.detach(),
        "loss_dyn": loss_dyn.detach(),
        "loss_terminal": loss_terminal.detach(),
        "velocity_norm": drift.norm(dim=1).mean(),
        "delta_s": delta_s.squeeze().detach(),
        "beta_t": beta_value(
            t, schedule=schedule, time_scale=time_scale, eps=eps
        ).detach(),
        "t": t.detach(),
    }


@torch.no_grad()
def evaluate_flow_map(model, z: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    return torch.stack([model(z, time_batch(t, z.shape[0], z)) for t in times], dim=1)


def sample_wgf_flow_map(
    model,
    z: torch.Tensor,
    num_steps: int,
    time_power: float = 1.0,
):
    if time_power <= 0:
        raise ValueError("time_power must be positive")
    times = torch.linspace(0, 1, num_steps + 1, device=z.device, dtype=z.dtype)
    times = times.pow(time_power)
    x = z
    for t, s in zip(times[:-1], times[1:]):
        x = model(x, t, s)
    return x


def endpoint_time_derivative(model, x: torch.Tensor, t, s):
    s = torch.as_tensor(s, device=x.device, dtype=x.dtype)

    def transition(end_time):
        return model(x, t, end_time)

    return torch.autograd.functional.jvp(
        transition,
        s,
        torch.ones_like(s),
        create_graph=True,
    )


def fast_wgf_flow_map_loss(
    model,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    data: torch.Tensor,
    *,
    dt: float,
    schedule: str = "constant",
    time_scale: float = 1.0,
    eps: float = 1e-3,
    path_weight: float = 0.1,
    lambda_wgf: float = 1.0,
    lambda_lagrangian: float = 1.0,
    lambda_eulerian: float = 0.0,
    lambda_sg: float = 0.0,
    lambda_terminal_sg: float = 0.0,
    lambda_boundary: float = 0.0,
    lambda_boundary_origin: float | None = None,
    lambda_boundary_arbitrary: float | None = None,
    boundary_step_size: float = 1.0,
    distill_max_span: float = 1.0,
    time_groups: int = 1,
    velocity_kind: str = "mmd",
    bandwidth=None,
    rbf_geometry: str = "raw",
    global_kde_support: bool = False,
    synchronize_times: bool = False,
):
    """Train a finite-difference WGF transition map.

    ``lambda_sg`` weights the global canonical-path identity
    ``Phi(Phi(z, 0, a), a, b) = Phi(z, 0, b)``.  Its target branch is detached,
    so this is a semi-gradient consistency update rather than a symmetric
    regression through both copies of the model.  ``lambda_terminal_sg``
    applies the same identity specifically at ``b=1``.  This directly
    supervises the last composed sampling hop without taking an extra WGF
    boundary step.
    """
    if time_groups < 1 or z.shape[0] % time_groups:
        raise ValueError("time_groups must be positive and divide the batch size")

    boundary_origin_weight = (
        lambda_boundary if lambda_boundary_origin is None else lambda_boundary_origin
    )
    boundary_arbitrary_weight = (
        lambda_boundary
        if lambda_boundary_arbitrary is None
        else lambda_boundary_arbitrary
    )
    has_boundary_loss = bool(boundary_origin_weight or boundary_arbitrary_weight)

    batch_size = z.shape[0]
    t0 = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
    dt = torch.as_tensor(float(dt), device=z.device, dtype=z.dtype)
    boundary_step_size = torch.as_tensor(
        float(boundary_step_size),
        device=z.device,
        dtype=z.dtype,
    )
    a = sample_grouped_times(
        batch_size,
        time_groups,
        z,
        synchronize=synchronize_times,
    )
    a = a * (1.0 - 2.0 * dt)
    a_eps = a + dt
    delta_t = transition_delta(
        a, a_eps, schedule=schedule, time_scale=time_scale, eps=eps
    ).reshape(-1, 1)
    max_span = torch.as_tensor(
        float(distill_max_span), device=z.device, dtype=z.dtype
    ).clamp(1e-6, 1.0)
    b_fraction = synchronize_distributed_tensor(
        torch.rand_like(a),
        enabled=synchronize_times,
    )
    b = a_eps + torch.minimum(1.0 - dt - a_eps, max_span).clamp_min(0.0) * b_fraction
    b_eps = b + dt

    with torch.no_grad():
        x_a = model(z, t0, a)
        if has_boundary_loss:
            boundary_a = sample_grouped_times(
                batch_size,
                1,
                z,
                synchronize=synchronize_times,
            )
            boundary_x_a = model(z, t0, boundary_a)
            boundary_ref_x_a = model(z_ref, t0, boundary_a)
            boundary_ref_1 = model(z_ref, t0, torch.ones_like(t0))
            boundary_ref_1_from_a = model(
                boundary_ref_x_a,
                boundary_a,
                torch.ones_like(boundary_a),
            )
        teacher_inputs = [z_ref]
        teacher_end_times = [a]
        if lambda_sg:
            teacher_inputs.append(z)
            teacher_end_times.append(b)
        teacher_outputs = model(
            torch.cat(teacher_inputs),
            t0.repeat(len(teacher_inputs)),
            torch.cat(teacher_end_times),
        ).chunk(len(teacher_inputs))
        ref_a = teacher_outputs[0]
        data_support = gather_global_support(data) if global_kde_support else data
        ref_a_support = (
            gather_global_support(ref_a, time_groups=time_groups)
            if global_kde_support
            else ref_a
        )
        if has_boundary_loss:
            boundary_ref_1_support = (
                gather_global_support(boundary_ref_1)
                if global_kde_support
                else boundary_ref_1
            )
            boundary_ref_1_from_a_support = (
                gather_global_support(boundary_ref_1_from_a)
                if global_kde_support
                else boundary_ref_1_from_a
            )
        velocity = grouped_wgf_velocity(
            velocity_kind,
            x_a,
            data_support,
            ref_a_support,
            bandwidth,
            time_groups,
            rbf_geometry=rbf_geometry,
        )
        kde_diagnostics = grouped_wgf_velocity_diagnostics(
            velocity_kind,
            x_a,
            data_support,
            ref_a_support,
            bandwidth,
            time_groups,
            rbf_geometry=rbf_geometry,
            max_queries_per_group=1,
        )

    student_inputs = [z, x_a, x_a, x_a]
    student_start_times = [t0, a, a, a]
    student_end_times = [a_eps, a_eps, b, b_eps]
    boundary_origin_index = None
    boundary_arbitrary_index = None
    terminal_semigroup_index = None
    if has_boundary_loss:
        boundary_origin_index = len(student_inputs)
        student_inputs.append(z)
        student_start_times.append(t0)
        student_end_times.append(torch.ones_like(t0))
        boundary_arbitrary_index = len(student_inputs)
        student_inputs.append(boundary_x_a)
        student_start_times.append(boundary_a)
        student_end_times.append(torch.ones_like(boundary_a))
    elif lambda_terminal_sg:
        # A direct endpoint is needed as the detached terminal-consistency target.
        boundary_origin_index = len(student_inputs)
        student_inputs.append(z)
        student_start_times.append(t0)
        student_end_times.append(torch.ones_like(t0))
    if lambda_terminal_sg:
        # Reuse the four grouped WGF path times, so the terminal segment is
        # constrained across the same time coverage as the main objective.
        terminal_semigroup_index = len(student_inputs)
        student_inputs.append(x_a)
        student_start_times.append(a)
        student_end_times.append(torch.ones_like(a))
    student_outputs = model(
        torch.cat(student_inputs),
        torch.cat(student_start_times),
        torch.cat(student_end_times),
    ).chunk(len(student_inputs))
    x_next, x_next_local, x_b_direct, x_b_eps = student_outputs[:4]
    loss_interpolant = F.mse_loss((x_next - x_a) / delta_t, velocity)
    loss_local_transition = F.mse_loss((x_next_local - x_a) / delta_t, velocity)
    loss_wgf = loss_interpolant + loss_local_transition
    loss_sg = (
        F.mse_loss(x_b_direct, teacher_outputs[1])
        if lambda_sg
        else torch.zeros((), device=z.device, dtype=z.dtype)
    )
    with torch.no_grad():
        x_b_local = model(x_b_direct.detach(), b, b_eps)
    loss_lagrangian = F.mse_loss(
        x_b_eps - x_b_direct,
        x_b_local - x_b_direct.detach(),
    )
    loss_eulerian = torch.zeros((), device=z.device, dtype=z.dtype)
    if lambda_eulerian:
        x_b_shifted = model(x_next_local, a_eps, b)
        loss_eulerian = F.mse_loss(x_b_shifted, x_b_direct.detach())
    loss_boundary_origin = torch.zeros((), device=z.device, dtype=z.dtype)
    loss_boundary_arbitrary = torch.zeros((), device=z.device, dtype=z.dtype)
    loss_terminal_sg = torch.zeros((), device=z.device, dtype=z.dtype)
    boundary_origin_velocity_norm = torch.zeros((), device=z.device, dtype=z.dtype)
    boundary_arbitrary_velocity_norm = torch.zeros((), device=z.device, dtype=z.dtype)
    if lambda_terminal_sg:
        x_1_direct = student_outputs[boundary_origin_index]
        x_1_terminal_composed = student_outputs[terminal_semigroup_index]
        loss_terminal_sg = F.mse_loss(x_1_terminal_composed, x_1_direct.detach())
    if has_boundary_loss:
        x_1 = student_outputs[boundary_origin_index]
        x_1_from_a = student_outputs[boundary_arbitrary_index]
        with torch.no_grad():
            boundary_origin_velocity = _configured_wgf_velocity(
                velocity_kind,
                x_1.detach(),
                data_support,
                boundary_ref_1_support,
                bandwidth,
                rbf_geometry=rbf_geometry,
            )
            boundary_arbitrary_velocity = _configured_wgf_velocity(
                velocity_kind,
                x_1_from_a.detach(),
                data_support,
                boundary_ref_1_from_a_support,
                bandwidth,
                rbf_geometry=rbf_geometry,
            )
            target_1 = x_1.detach() + boundary_step_size * boundary_origin_velocity
            target_1_from_a = (
                x_1_from_a.detach() + boundary_step_size * boundary_arbitrary_velocity
            )
        loss_boundary_origin = F.mse_loss(x_1, target_1)
        loss_boundary_arbitrary = F.mse_loss(x_1_from_a, target_1_from_a)
        boundary_origin_velocity_norm = boundary_origin_velocity.norm(dim=1).mean()
        boundary_arbitrary_velocity_norm = boundary_arbitrary_velocity.norm(
            dim=1
        ).mean()
    total = (
        path_weight
        * (
            lambda_wgf * loss_wgf
            + lambda_lagrangian * loss_lagrangian
            + lambda_eulerian * loss_eulerian
        )
        + lambda_sg * loss_sg
        + lambda_terminal_sg * loss_terminal_sg
        + boundary_origin_weight * loss_boundary_origin
        + boundary_arbitrary_weight * loss_boundary_arbitrary
    )

    with torch.no_grad():
        prediction = (x_next - x_a) / delta_t
        local_prediction = (x_next_local - x_a) / delta_t
        velocity_rms = velocity.square().mean().sqrt()

    stats = {
        "loss": total.detach(),
        "loss_wgf": loss_wgf.detach(),
        "loss_interpolant": loss_interpolant.detach(),
        "loss_local_transition": loss_local_transition.detach(),
        "loss_lagrangian": loss_lagrangian.detach(),
        "loss_eulerian": loss_eulerian.detach(),
        "path_weight": path_weight,
        "lambda_wgf": lambda_wgf,
        "lambda_terminal_semigroup": lambda_terminal_sg,
        "lambda_boundary_origin": boundary_origin_weight,
        "lambda_boundary_arbitrary": boundary_arbitrary_weight,
        "boundary_step_size": boundary_step_size.detach(),
        "loss_boundary_origin": loss_boundary_origin.detach(),
        "loss_boundary_arbitrary": loss_boundary_arbitrary.detach(),
        "loss_terminal_semigroup": loss_terminal_sg.detach(),
        "velocity_norm": velocity.norm(dim=1).mean(),
        "velocity_rms": velocity_rms,
        "origin_prediction_norm": prediction.norm(dim=1).mean(),
        "local_prediction_norm": local_prediction.norm(dim=1).mean(),
        "origin_velocity_cosine": F.cosine_similarity(
            prediction, velocity, dim=1
        ).mean(),
        "local_velocity_cosine": F.cosine_similarity(
            local_prediction, velocity, dim=1
        ).mean(),
        "boundary_origin_velocity_norm": boundary_origin_velocity_norm,
        "boundary_arbitrary_velocity_norm": boundary_arbitrary_velocity_norm,
        "boundary_start_time": (
            boundary_a[0].detach()
            if has_boundary_loss
            else (
                a[0].detach()
                if lambda_terminal_sg
                else torch.zeros((), device=z.device, dtype=z.dtype)
            )
        ),
        "origin_residual_ratio": (
            (prediction - velocity).square().mean().sqrt()
            / velocity_rms.clamp_min(1e-8)
        ),
        "local_residual_ratio": (
            (local_prediction - velocity).square().mean().sqrt()
            / velocity_rms.clamp_min(1e-8)
        ),
        "path_sample_mean": x_a.mean(),
        "path_sample_std": x_a.std(),
        "path_time_mean": a.mean().detach(),
        "path_time_std": a.std().detach(),
        "distill_max_span": max_span.detach(),
        "distill_span_mean": (b - a).mean().detach(),
        "time_groups": time_groups,
        **{f"kde_{key}": value for key, value in kde_diagnostics.items()},
    }
    if lambda_sg:
        stats["loss_sg"] = loss_sg.detach()
        stats["loss_semigroup"] = loss_sg.detach()
    return total, stats


def wgf_flow_map_loss(
    model,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    data: torch.Tensor,
    *,
    dt: float | None = None,
    eps_t: float | None = None,
    schedule: str = "constant",
    time_scale: float = 1.0,
    eps: float = 1e-3,
    lambda_sg: float = 0.1,
    semigroup_composition_weight: float = 0.0,
    lambda_terminal: float = 0.0,
    terminal_steps: int = 1,
    terminal_time_power: float = 1.0,
    distill_max_span: float = 1.0,
    time_groups: int = 1,
    path_weight: float = 1.0,
    velocity_kind: str = "mmd",
    bandwidth=None,
    rbf_geometry: str = "raw",
):
    if dt is None:
        if eps_t is None:
            raise TypeError("wgf_flow_map_loss requires dt or eps_t")
        dt = eps_t

    t0 = torch.zeros((), device=z.device, dtype=z.dtype)
    dt = torch.as_tensor(float(dt), device=z.device, dtype=z.dtype)
    if time_groups < 1 or z.shape[0] % time_groups:
        raise ValueError("time_groups must be positive and divide the batch size")

    def grouped_times():
        if time_groups == 1:
            return torch.rand((), device=z.device, dtype=z.dtype)
        return torch.rand(
            time_groups, device=z.device, dtype=z.dtype
        ).repeat_interleave(z.shape[0] // time_groups)

    t = grouped_times() * (1.0 - dt)
    t_eps = t + dt
    delta_t = transition_delta(
        t, t_eps, schedule=schedule, time_scale=time_scale, eps=eps
    ).reshape(-1, 1)
    max_span = torch.as_tensor(
        float(distill_max_span), device=z.device, dtype=z.dtype
    ).clamp(1e-6, 1.0)

    # Origin path loss (0 -> t)
    with torch.no_grad():
        x_t = model(z, t0, t)
        ref_t = model(z_ref, t0, t)
        velocity = grouped_wgf_velocity(
            velocity_kind,
            x_t,
            data,
            ref_t,
            bandwidth,
            time_groups,
            rbf_geometry=rbf_geometry,
        )
    x_next = model(z, t0, t_eps)
    loss_interpolant = F.mse_loss((x_next - x_t) / delta_t, velocity)
    x_next_local = model(x_t.detach(), t, t_eps)
    loss_local_transition = F.mse_loss((x_next_local - x_t) / delta_t, velocity)

    # Diagonal velocity matching, preconditioned to displacement scale.
    with torch.no_grad():
        target_diag_step = delta_t * velocity
    _, diag_velocity = endpoint_time_derivative(model, x_t, t, t)
    loss_diagonal = F.mse_loss(dt * diag_velocity, target_diag_step)

    # Lagrangian self-distillation: d/db Phi(x_a, a, b) = v_diag(Phi(x_a, a, b), b).
    a = grouped_times()
    b = a + torch.minimum(1.0 - a, max_span) * grouped_times()
    with torch.no_grad():
        x_a = model(z, t0, a)
    x_b, lagrangian_velocity = endpoint_time_derivative(model, x_a, a, b)
    with torch.no_grad():
        _, target_lagrangian_velocity = endpoint_time_derivative(
            model, x_b.detach(), b, b
        )
    # torch.autograd.functional.jvp(create_graph=True) re-enables autograd even
    # inside no_grad; detach explicitly so this branch remains a frozen teacher.
    target_lagrangian_velocity = target_lagrangian_velocity.detach()
    loss_lagrangian = F.mse_loss(
        dt * lagrangian_velocity, dt * target_lagrangian_velocity
    )

    # Path anchor plus optional arbitrary-triple PSD consistency.
    loss_sg = torch.zeros((), device=z.device, dtype=z.dtype)
    loss_sg_anchor = torch.zeros((), device=z.device, dtype=z.dtype)
    loss_sg_composition = torch.zeros((), device=z.device, dtype=z.dtype)
    semigroup_error = torch.zeros((), device=z.device, dtype=z.dtype)
    if lambda_sg:
        s_sg = grouped_times()
        t_sg = s_sg + torch.minimum(1.0 - s_sg, max_span) * grouped_times()
        r_sg = s_sg + (t_sg - s_sg) * grouped_times()

        with torch.no_grad():
            x_s_path = model(z, t0, s_sg)
            x_t_path = model(z, t0, t_sg)

        x_t_direct = model(x_s_path, s_sg, t_sg)
        if semigroup_composition_weight:
            x_r_composed = model(x_s_path.detach(), s_sg, r_sg)
            x_t_composed = model(x_r_composed, r_sg, t_sg)
        else:
            with torch.no_grad():
                x_r_composed = model(x_s_path, s_sg, r_sg)
                x_t_composed = model(x_r_composed, r_sg, t_sg)
        loss_sg_anchor = F.mse_loss(x_t_direct, x_t_path)
        loss_sg_composition = F.mse_loss(x_t_composed, x_t_direct.detach())
        loss_sg = loss_sg_anchor + semigroup_composition_weight * loss_sg_composition
        semigroup_error = (x_t_direct.detach() - x_t_composed).norm(dim=1).mean()

    loss_terminal = torch.zeros((), device=z.device, dtype=z.dtype)
    terminal_velocity_norm = torch.zeros((), device=z.device, dtype=z.dtype)
    terminal_sample_mean = torch.zeros((), device=z.device, dtype=z.dtype)
    terminal_sample_std = torch.zeros((), device=z.device, dtype=z.dtype)
    if lambda_terminal:
        if terminal_steps <= 1:
            x_1 = model(z, t0, torch.ones((), device=z.device, dtype=z.dtype))
            with torch.no_grad():
                ref_1 = model(z_ref, t0, torch.ones((), device=z.device, dtype=z.dtype))
        else:
            x_1 = sample_wgf_flow_map(model, z, terminal_steps, terminal_time_power)
            with torch.no_grad():
                ref_1 = sample_wgf_flow_map(
                    model, z_ref, terminal_steps, terminal_time_power
                )
        with torch.no_grad():
            terminal_velocity = _configured_wgf_velocity(
                velocity_kind,
                x_1.detach(),
                data,
                ref_1,
                bandwidth,
                rbf_geometry=rbf_geometry,
            )
            target_1 = x_1.detach() + terminal_velocity
        loss_terminal = F.mse_loss(x_1, target_1)
        terminal_velocity_norm = terminal_velocity.norm(dim=1).mean()
        terminal_sample_mean = x_1.detach().mean()
        terminal_sample_std = x_1.detach().std()

    loss_wgf = (
        loss_interpolant + loss_local_transition + loss_diagonal + loss_lagrangian
    )
    total = (
        path_weight * loss_wgf + lambda_sg * loss_sg + lambda_terminal * loss_terminal
    )

    with torch.no_grad():
        origin_prediction = (x_next - x_t) / delta_t
        local_prediction = (x_next_local - x_t) / delta_t
        velocity_rms = velocity.square().mean().sqrt()
        velocity_max = velocity.norm(dim=1).max()
        origin_residual_ratio = (
            origin_prediction - velocity
        ).square().mean().sqrt() / velocity_rms.clamp_min(1e-8)
        local_residual_ratio = (
            local_prediction - velocity
        ).square().mean().sqrt() / velocity_rms.clamp_min(1e-8)

    return total, {
        "loss": total.detach(),
        "loss_wgf": loss_wgf.detach(),
        "path_weight": path_weight,
        "loss_interpolant": loss_interpolant.detach(),
        "loss_local_transition": loss_local_transition.detach(),
        "loss_diagonal": loss_diagonal.detach(),
        "loss_lagrangian": loss_lagrangian.detach(),
        "loss_sg": loss_sg.detach(),
        "loss_sg_anchor": loss_sg_anchor.detach(),
        "loss_sg_composition": loss_sg_composition.detach(),
        "loss_terminal": loss_terminal.detach(),
        "velocity_norm": velocity.norm(dim=1).mean(),
        "velocity_rms": velocity_rms,
        "velocity_max": velocity_max,
        "origin_residual_ratio": origin_residual_ratio,
        "local_residual_ratio": local_residual_ratio,
        "path_sample_mean": x_t.mean(),
        "path_sample_std": x_t.std(),
        "terminal_velocity_norm": terminal_velocity_norm.detach(),
        "terminal_sample_mean": terminal_sample_mean.detach(),
        "terminal_sample_std": terminal_sample_std.detach(),
        "semigroup_error": semigroup_error.detach(),
        "delta_t": delta_t.mean().detach(),
        "distill_max_span": max_span.detach(),
        "path_time_mean": t.mean().detach(),
        "path_time_std": t.std().detach() if t.ndim else torch.zeros_like(t),
        "lagrangian_start_time_mean": a.mean().detach(),
        "lagrangian_span_mean": (b - a).mean().detach(),
        "time_groups": time_groups,
    }
