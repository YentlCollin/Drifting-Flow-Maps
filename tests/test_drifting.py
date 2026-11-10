import math

import pytest
import torch
from flowmaps.objectives.drifting import (
    feature_map_to_groups,
    official_drifting_loss,
    raw_latent_pyramid_features,
    summed_feature_drifting_loss,
)


def test_loss_is_finite_for_zero_initialized_generator_and_has_gradient():
    generated = torch.zeros(3, 5, 4, requires_grad=True)
    positive = torch.randn(3, 7, 4, generator=torch.Generator().manual_seed(4))
    loss, info = official_drifting_loss(generated, positive)
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()
    assert set(info) == {
        "scale",
        "force_energy_0.02",
        "force_energy_0.05",
        "force_energy_0.2",
    }
    loss.mean().backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert generated.grad.abs().sum() > 0
    assert positive.grad is None


def test_feature_scale_is_removed_like_the_release():
    generator = torch.Generator().manual_seed(9)
    generated = torch.randn(2, 6, 3, generator=generator)
    positive = torch.randn(2, 8, 3, generator=generator)
    base, _ = official_drifting_loss(generated, positive)
    scaled, _ = official_drifting_loss(17.0 * generated, 17.0 * positive)
    torch.testing.assert_close(base, scaled, rtol=2e-5, atol=2e-5)


def test_one_temperature_matches_direct_release_equations():
    generator = torch.Generator().manual_seed(17)
    generated = torch.randn(1, 3, 2, generator=generator, requires_grad=True)
    positive = torch.randn(1, 4, 2, generator=generator)
    actual, actual_info = official_drifting_loss(
        generated,
        positive,
        temperatures=(0.2,),
    )

    old = generated.detach()
    targets = torch.cat((old, positive), dim=1)
    dot = torch.einsum("gnd,gmd->gnm", old, targets)
    squared = (
        old.square().sum(-1)[:, :, None]
        + targets.square().sum(-1)[:, None, :]
        - 2 * dot
    )
    distance = squared.clamp_min(1e-8).sqrt()
    scale = distance.mean()
    scale_inputs = (scale / math.sqrt(2)).clamp_min(1e-3)
    normalized = distance / scale.clamp_min(1e-3)
    normalized[:, :, :3].add_(torch.eye(3) * 100.0)
    logits = -normalized / 0.2
    affinity = (logits.softmax(-1) * logits.softmax(-2)).clamp_min(1e-6).sqrt()
    neg, pos = affinity[:, :, :3], affinity[:, :, 3:]
    coeff = torch.cat(
        (-neg * pos.sum(-1, keepdim=True), pos * neg.sum(-1, keepdim=True)), dim=-1
    )
    target_scaled = targets / scale_inputs
    old_scaled = old / scale_inputs
    force = torch.einsum("gnm,gmd->gnd", coeff, target_scaled)
    force = force - coeff.sum(-1, keepdim=True) * old_scaled
    energy = force.square().mean()
    goal = old_scaled + force / energy.clamp_min(1e-8).sqrt()
    expected = (generated / scale_inputs - goal).square().mean((-1, -2))

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_info["scale"], scale)
    torch.testing.assert_close(actual_info["force_energy_0.2"], energy)


def test_feature_helpers_and_sum_preserve_expected_layout():
    generated_map = torch.randn(5, 4, 2, 3, requires_grad=True)
    positive_map = torch.randn(7, 4, 2, 3)
    groups = feature_map_to_groups(generated_map)
    assert groups.shape == (6, 5, 4)

    generated = {
        "local": generated_map.flatten(2).transpose(1, 2),
        "global": generated_map.flatten(1).unsqueeze(1),
    }
    positive = {
        "local": positive_map.flatten(2).transpose(1, 2),
        "global": positive_map.flatten(1).unsqueeze(1),
    }
    loss, stats = summed_feature_drifting_loss(generated, positive)
    assert loss.ndim == 0
    assert set(key for key in stats if key.startswith("loss/")) == {
        "loss/local",
        "loss/global",
    }
    loss.backward()
    assert generated_map.grad is not None


def test_summed_feature_loss_accepts_detached_remote_generated_support():
    generator = torch.Generator().manual_seed(29)
    generated = {
        "tokens": torch.randn(4, 3, 5, generator=generator, requires_grad=True)
    }
    positive = {"tokens": torch.randn(8, 3, 5, generator=generator)}
    remote = {"tokens": torch.randn(4, 3, 5, generator=generator)}

    loss, _ = summed_feature_drifting_loss(
        generated,
        positive,
        fixed_negative_features=remote,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert generated["tokens"].grad is not None
    assert torch.isfinite(generated["tokens"].grad).all()


def test_summed_feature_loss_rejects_mismatched_remote_feature_keys():
    generated = {"tokens": torch.zeros(2, 1, 3)}
    positive = {"tokens": torch.ones(2, 1, 3)}
    with pytest.raises(ValueError, match="fixed-negative"):
        summed_feature_drifting_loss(
            generated,
            positive,
            fixed_negative_features={"different": torch.zeros(2, 1, 3)},
        )


def test_raw_latent_pyramid_has_multiscale_tokens_and_gradients():
    latent = torch.randn(3, 256, requires_grad=True)
    features = raw_latent_pyramid_features(latent)
    assert {name: tuple(value.shape) for name, value in features.items()} == {
        "global": (3, 1, 256),
        "latent_spatial": (3, 64, 4),
        "latent_channel_rms": (3, 1, 4),
        "latent_mean_2": (3, 16, 4),
        "latent_std_2": (3, 16, 4),
        "latent_mean_4": (3, 4, 4),
        "latent_std_4": (3, 4, 4),
    }
    sum(value.square().mean() for value in features.values()).backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


@pytest.mark.parametrize("temperature", [(), (0.0,), (float("nan"),)])
def test_invalid_temperatures_are_rejected(temperature):
    with pytest.raises(ValueError, match="temperatures"):
        official_drifting_loss(
            torch.zeros(1, 2, 3), torch.ones(1, 2, 3), temperatures=temperature
        )
