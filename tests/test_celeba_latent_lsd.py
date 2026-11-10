import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

import celeba_latent_lsd as runner
from src.celeba_latent import (
    CELEBA_IMAGE_SHAPE,
    CELEBA_LATENT_SHAPE,
    LATENT_CACHE_FORMAT,
    LATENT_CACHE_VERSION,
    TransitionConditionedLatentUNet,
)
from src.celeba_latent_lsd import (
    CelebALatentLSDBackbone,
    CelebALatentLSDTrainingModel,
    project_log_variance_weight_,
)
from src.flow_map_self_distillation import (
    MagnitudePreservingTimeLogVariance,
    ResidualTwoTimeFlowMap,
    endpoint_time_jvp,
)


def write_fake_cache(path: Path, count: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    latents = torch.randn(count, *CELEBA_LATENT_SHAPE, generator=generator)
    metadata = {
        "format": LATENT_CACHE_FORMAT,
        "version": LATENT_CACHE_VERSION,
        "model_id": "unit-test-vae",
        "scaling_factor": 0.18215,
        "input_range": "zero_one",
        "encoding": "posterior_mean",
        "image_shape": list(CELEBA_IMAGE_SHAPE),
        "latent_shape": list(CELEBA_LATENT_SHAPE),
        "count": count,
        "dtype": str(latents.dtype),
    }
    torch.save({"latents": latents, "metadata": metadata}, path)
    return latents


def tiny_training_model() -> CelebALatentLSDTrainingModel:
    return CelebALatentLSDTrainingModel(
        ResidualTwoTimeFlowMap(
            CelebALatentLSDBackbone(
                base_channels=8,
                time_embedding_dim=8,
                condition_dim=16,
                sigma_data=0.5,
            ),
            MagnitudePreservingTimeLogVariance(8),
        )
    )


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_backbone_exposes_absolute_phi_and_nonsingular_start_delta_conditioning():
    torch.manual_seed(1)
    backbone = CelebALatentLSDBackbone(
        base_channels=8,
        time_embedding_dim=8,
        condition_dim=16,
        sigma_data=0.5,
    )
    flow = ResidualTwoTimeFlowMap(backbone)
    x = torch.randn(2, 256)
    start = torch.tensor([0.999, 1.0])
    end = torch.tensor([1.0, 1.0])

    velocity = backbone.endpoint(x, start, end)
    mapped = flow(x, start, end)
    _, derivative = endpoint_time_jvp(flow, x, start, end, create_graph=True)

    assert backbone.endpoint_parameterization == "absolute"
    assert backbone.time_conditioning == "start_delta_v1"
    torch.testing.assert_close(mapped, x + (end - start)[:, None] * velocity)
    assert torch.isfinite(derivative).all()
    derivative.square().mean().backward()
    assert all(parameter.grad is not None for parameter in backbone.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in backbone.parameters())


def test_sigma_data_preconditions_input_and_output_separately_from_source_prior():
    class EchoBackbone(CelebALatentLSDBackbone):
        def _raw_network(self, spatial, condition):
            self.observed = spatial.detach().clone()
            return spatial

    backbone = EchoBackbone(
        base_channels=4,
        time_embedding_dim=4,
        condition_dim=8,
        sigma_data=0.5,
    )
    x = torch.randn(2, 256)
    velocity = backbone.endpoint(x, 0.2, 0.8)

    torch.testing.assert_close(backbone.observed.flatten(1), 2.0 * x)
    torch.testing.assert_close(velocity, x)


def test_time_embedding_matches_released_mp_formula_without_two_pi_factor():
    backbone = CelebALatentLSDBackbone(
        base_channels=4,
        time_embedding_dim=8,
        condition_dim=8,
    )
    times = torch.tensor([0.0, 0.25, 1.0])
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(4, dtype=torch.float32) / 4
    )
    angles = times[:, None] * frequencies[None, :]
    expected = math.sqrt(2.0) * torch.cat((angles.cos(), angles.sin()), dim=1)

    torch.testing.assert_close(backbone.time_embedding(times), expected)


def test_exact_75_25_loss_has_gradients_for_backbone_and_logvariance():
    torch.manual_seed(2)
    model = tiny_training_model()
    x0 = torch.randn(8, 256)
    x1 = torch.randn(8, 256)

    loss, stats = model(x0, x1, model.flow_map)
    loss.backward()

    assert stats["diagonal_count"] == 6
    assert stats["offdiagonal_count"] == 2
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert sum(parameter.grad.abs().sum() for parameter in model.parameters()) > 0


def test_zero_phi_initialization_makes_every_map_identity_and_diagonal_is_bitwise():
    torch.manual_seed(21)
    backbone = CelebALatentLSDBackbone(
        base_channels=8,
        time_embedding_dim=8,
        condition_dim=16,
        zero_init_velocity=True,
    )
    flow = ResidualTwoTimeFlowMap(backbone)
    x = torch.randn(4, 256)
    start = torch.tensor([0.0, 0.2, 0.7, 1.0])
    end = torch.tensor([1.0, 0.9, 0.71, 1.0])

    assert torch.equal(backbone.endpoint(x, start, end), torch.zeros_like(x))
    assert torch.equal(flow(x, start, end), x)
    assert torch.equal(flow(x, start, start), x)

    model = CelebALatentLSDTrainingModel(
        ResidualTwoTimeFlowMap(backbone, MagnitudePreservingTimeLogVariance(8))
    )
    loss, _ = model(torch.randn_like(x), torch.randn_like(x), model.flow_map)
    loss.backward()
    assert sum(parameter.grad.abs().sum() for parameter in model.parameters()) > 0


def test_logvariance_projection_matches_edm2_unit_magnitude_rule():
    model = tiny_training_model()
    head = model.flow_map.log_variance_network
    with torch.no_grad():
        head.weight.normal_(mean=0.0, std=7.0)
        before = head.weight.detach().clone()

    count = project_log_variance_weight_(model)
    rms = head.weight.float().square().mean(dim=1).sqrt()

    assert count == 1
    assert not torch.equal(before, head.weight)
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=2e-4, atol=2e-4)


def test_cli_defaults_distinguish_latent_prior_from_network_preconditioner():
    args = runner.parse_args([])

    assert args.source_std == 1.0
    assert args.sigma_data == 0.5
    assert args.coupling == "independent"
    assert args.lsd_target == "current"
    assert args.global_batch == 256
    assert args.steps == 800_000
    assert args.verify_gradients is False
    assert args.zero_init_velocity is True


def test_teacher_coupled_ablation_uses_same_source_and_frozen_endpoint():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "teacher.pt"
        teacher_source = TransitionConditionedLatentUNet(
            base_channels=4,
            time_embedding_dim=4,
            condition_dim=8,
            zero_init_endpoint=False,
            endpoint_parameterization="absolute",
        )
        torch.save(teacher_source.state_dict(), checkpoint)
        args = SimpleNamespace(
            coupling="teacher_coupled_ablation",
            gmd_teacher_checkpoint=checkpoint,
            gmd_teacher_weight_kind="ema",
            gmd_teacher_base_channels=4,
            gmd_teacher_time_embedding_dim=4,
            gmd_teacher_condition_dim=8,
            device=torch.device("cpu"),
        )
        teacher, metadata = runner.load_gmd_teacher(
            args,
            cache_sha256="a" * 64,
            train_start=0,
            train_end=8,
        )
        x0 = torch.randn(3, 256, requires_grad=True)
        targets = runner._teacher_coupled_targets(teacher, x0)
        times = torch.zeros(3)

        torch.testing.assert_close(targets, teacher_source(x0.detach(), times, torch.ones_like(times)))
        assert not targets.requires_grad
        assert metadata["objective_attested"] is False
        assert metadata["coupling"].startswith("x1=stopgrad")


def test_single_process_smoke_exports_full_provenance_and_resumes_exactly():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache = root / "cache.pt"
        write_fake_cache(cache)
        continuous = root / "continuous"
        resumed = root / "resumed"
        common = [
            "--smoke",
            "--cache",
            str(cache),
            "--global-batch",
            "4",
            "--checkpoint-every",
            "1",
        ]

        runner.main([*common, "--steps", "2", "--run-dir", str(continuous)])
        runner.main([*common, "--steps", "1", "--run-dir", str(resumed)])
        runner.main(
            [
                *common,
                "--steps",
                "2",
                "--resume",
                str(resumed / "checkpoint_final.pt"),
                "--run-dir",
                str(resumed),
            ]
        )

        continuous_checkpoint = torch.load(
            continuous / "checkpoint_final.pt", map_location="cpu", weights_only=True
        )
        resumed_checkpoint = torch.load(
            resumed / "checkpoint_final.pt", map_location="cpu", weights_only=True
        )
        assert continuous_checkpoint["completed_step"] == 2
        assert resumed_checkpoint["completed_step"] == 2
        assert_state_equal(continuous_checkpoint["model"], resumed_checkpoint["model"])
        assert_state_equal(continuous_checkpoint["ema"], resumed_checkpoint["ema"])

        for path in (
            continuous / "config.json",
            continuous / "provenance.json",
            continuous / "source_snapshot" / "manifest.json",
            continuous / "model_raw_final.pt",
            continuous / "model_ema_final.pt",
        ):
            assert path.is_file(), path
        config = json.loads((continuous / "config.json").read_text())
        provenance = json.loads((continuous / "provenance.json").read_text())
        signature = continuous_checkpoint["training_signature"]
        assert signature["diagonal_count_local"] == 3
        assert signature["offdiagonal_count_local"] == 1
        assert signature["lsd_target"] == "current"
        assert signature["ddp_broadcast_buffers"] is False
        assert config["adaptation_notice"].startswith("LSD objective port")
        assert provenance["edm2_projection"]["scope"].endswith("only")
        assert provenance["ddp"]["broadcast_buffers"] is False
        assert list(resumed.glob("config_resume_*.json"))
        assert list(resumed.glob("source_snapshot_resume_*"))


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_torchrun_gloo_two_rank_smoke_covers_jvp_backward_and_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache = root / "cache.pt"
        run_dir = root / "ddp"
        write_fake_cache(cache, count=32)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(runner.__file__).resolve()),
            "--smoke",
            "--steps",
            "2",
            "--global-batch",
            "8",
            "--cache",
            str(cache),
            "--run-dir",
            str(run_dir),
        ]
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(
            command,
            cwd=Path(runner.__file__).resolve().parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

        checkpoint = torch.load(
            run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=True
        )
        signature = checkpoint["training_signature"]
        assert checkpoint["completed_step"] == 2
        assert signature["world_size"] == 2
        assert signature["global_batch"] == 8
        assert signature["local_batch"] == 4
        assert signature["ddp_broadcast_buffers"] is False
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        records = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        ]
        assert len([record for record in records if record["event"] == "train"]) == 2
