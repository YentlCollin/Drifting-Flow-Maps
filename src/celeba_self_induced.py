"""Self-induced feature-drifting signals, without terminal rollout training.

The W/L/E/B0/Ba equations follow ``fast_wgf_flow_map_loss`` and the 36-cell
MNIST signal ablation. S is the canonical composition from main.tex. W uses
the frozen MAE/ConvNeXt normalized GMD force instead of raw-pixel KL-KDE.
This is an adaptation of the directional teacher, not an exact WGF solver.
"""

from dataclasses import dataclass, asdict
import itertools
import math

import torch
from torch import nn

from src.celeba_gmd_composed_rollout import ComposedRollout


FACTORS = ("path", "lagrangian", "eulerian", "composition", "boundary_origin", "boundary_arbitrary")


@dataclass(frozen=True)
class SignalWeights:
    path: float = 0.1
    lagrangian: float = 0.1
    eulerian: float = 10.0
    composition: float = 0.1
    boundary_origin: float = 1.0
    boundary_arbitrary: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("weights must be finite and nonnegative")

    def as_dict(self):
        return asdict(self)


def factorial_variants():
    """All 63 nonempty combinations; the all-zero cell has no training signal."""
    return {
        "".join(str(bit) for bit in bits): dict(zip(FACTORS, bits))
        for bits in itertools.product((0, 1), repeat=len(FACTORS)) if any(bits)
    }


def masked_weights(mask, scales=SignalWeights()):
    if mask not in factorial_variants():
        raise ValueError("mask must identify a nonempty six-factor cell")
    return SignalWeights(**{k: v * factorial_variants()[mask][k] for k, v in scales.as_dict().items()})


def sample_times(generator, *, dt=0.05, max_span=0.5):
    """Always consume three CPU draws, independent of the enabled losses.

    All ranks use this same generator seed/state. Every feature force uses
    one homogeneous time across its complete global source support.
    """
    if not 0 < dt < 0.5 or not 0 < max_span <= 1:
        raise ValueError("invalid finite-difference step or span")
    draws = torch.rand(3, generator=generator, dtype=torch.float64).tolist()
    a = draws[0] * (1 - 2 * dt)
    b = a + dt + draws[1] * min(1 - a - 2 * dt, max_span)
    boundary_a = draws[2] * (1 - dt)
    return a, b, boundary_a


class SelfInducedSignals(ComposedRollout):
    """One DDP forward containing direct transitions and detached teachers.

    Inheriting the serialization wrapper only reuses its artifact format;
    its rollout forward is never called here. No terminal loss differentiates
    through a chain: both B0 and Ba supervise a single transition.
    """

    def __init__(self, flow, weights, *, dt=0.05):
        super().__init__(flow)
        self.weights = weights
        self.dt = dt

    def forward(self, z, labels, a, b, boundary_a):
        w, dt = self.weights, self.dt
        if not (0 <= a < a + dt <= b < b + dt <= 1 and 0 <= boundary_a < 1):
            raise ValueError("time ordering must permit both finite differences")
        # no_grad calls must not populate autocast's parameter cache before
        # student/checkpoint recomputation calls; see continuous pilot fix.
        out = {}
        need_source = any((w.path, w.lagrangian, w.eulerian, w.composition))
        if need_source:
            with torch.no_grad():
                source = self.flow(z, labels, 0.0, a).detach()
            out["source"] = source
        if w.path or w.eulerian:
            local = self.flow(source, labels, a, a + dt)
            if w.path:
                out["origin"] = self.flow(z, labels, 0.0, a + dt)
                out["local"] = local
        if w.lagrangian or w.eulerian or w.composition:
            at_b = self.flow(source, labels, a, b)
        if w.lagrangian:
            at_b_eps = self.flow(source, labels, a, b + dt)
            with torch.no_grad():
                local_b = self.flow(at_b.detach(), labels, b, b + dt)
                teacher_displacement = local_b - at_b.detach()
            # Historical MNIST finite-difference L, without an extra dt^-2.
            out["lagrangian"] = ((at_b_eps - at_b) - teacher_displacement).float().square().mean()
        if w.eulerian:
            shifted = self.flow(local, labels, a + dt, b)
            out["eulerian"] = (shifted.float() - at_b.detach().float()).square().mean()
        if w.composition:
            with torch.no_grad():
                teacher_b = self.flow(z, labels, 0.0, b)
            out["composition"] = (at_b.float() - teacher_b.float()).square().mean()
        if w.boundary_origin:
            out["boundary_origin"] = self.flow(z, labels, 0.0, 1.0)
        if w.boundary_arbitrary:
            with torch.no_grad():
                boundary_source = self.flow(z, labels, 0.0, boundary_a).detach()
            out["boundary_arbitrary"] = self.flow(boundary_source, labels, boundary_a, 1.0)
        return out


def signal_objective(outputs, weights, encoder, positives, negatives, *, force_step, temperatures):
    """Calculate only enabled losses; return separate unweighted components."""
    from src.celeba_gmd_local_two_view import global_sharded_mae_convnext_two_view_loss
    from src.celeba_gmd_sharded_features import global_sharded_mae_convnext_loss

    components = {}
    if weights.path:
        value, _ = global_sharded_mae_convnext_two_view_loss(
            encoder, outputs["source"], outputs["origin"], outputs["local"],
            positives, negatives, force_step=force_step, temperatures=temperatures,
        )
        # Existing helper returns a mean of two views; the manuscript sums them.
        components["path"] = 2 * value
    for name in ("boundary_origin", "boundary_arbitrary"):
        if getattr(weights, name):
            components[name], _ = global_sharded_mae_convnext_loss(
                encoder, outputs[name], positives, negatives, temperatures=temperatures,
            )
    for name in ("lagrangian", "eulerian", "composition"):
        if getattr(weights, name):
            components[name] = outputs[name]
    if not components:
        raise ValueError("no training objective enabled")
    total = sum(getattr(weights, name) * value for name, value in components.items())
    return total, components
