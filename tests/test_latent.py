import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import flowmaps.data.prepare_latents as cache_script
import torch
import torch.nn.functional as F
from flowmaps.models.latent import (
    CELEBA_LATENT_DIM,
    DEFAULT_POSTERIOR_SAMPLE_SEED,
    POSTERIOR_SAMPLE_INDEXING,
    POSTERIOR_SAMPLE_RNG_SCHEME,
    CelebALatentCodec,
    LatentGaussianPrior,
    TransitionConditionedLatentUNet,
    directory_file_hashes,
    encode_loader_to_cache,
    flatten_latents,
    flow_map_endpoint_weight,
    load_latent_cache,
    save_latent_cache,
    unflatten_latents,
)
from torch.utils.data import DataLoader, TensorDataset


def _well_conditioned_latents(mean: float = 0.25) -> torch.Tensor:
    basis = torch.eye(CELEBA_LATENT_DIM, dtype=torch.float32)
    return torch.cat((basis, -basis), dim=0).mul_(CELEBA_LATENT_DIM**0.5).add_(mean)


class _MockPosterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean
        self.std = torch.full_like(mean, 0.5)
        self.sample_calls = 0

    def sample(self, generator=None) -> torch.Tensor:
        self.sample_calls += 1
        if not isinstance(generator, list) or len(generator) != self.mean.shape[0]:
            raise AssertionError("posterior sampling requires one generator per image")
        noise = torch.cat(
            [
                torch.randn(
                    (1, *self.mean.shape[1:]),
                    generator=image_generator,
                    dtype=self.mean.dtype,
                )
                for image_generator in generator
            ],
            dim=0,
        ).to(self.mean.device)
        return self.mean + self.std * noise


class _MockVAE(torch.nn.Module):
    """Small CPU stand-in exposing the relevant AutoencoderKL interface."""

    def __init__(self, scaling_factor: float = 0.25) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(scaling_factor=scaling_factor)
        self.encode_calls = 0
        self.decode_calls = 0
        self.last_encode_input = None
        self.last_decode_input = None
        self.last_posterior = None

    def encode(self, images: torch.Tensor):
        self.encode_calls += 1
        self.last_encode_input = images.detach().clone()
        pooled = F.avg_pool2d(images, kernel_size=8)
        fourth_channel = pooled.mean(dim=1, keepdim=True)
        self.last_posterior = _MockPosterior(torch.cat([pooled, fourth_channel], dim=1))
        return SimpleNamespace(latent_dist=self.last_posterior)

    def decode(self, latents: torch.Tensor):
        self.decode_calls += 1
        self.last_decode_input = latents.detach().clone()
        images = F.interpolate(latents[:, :3], scale_factor=8, mode="nearest")
        return SimpleNamespace(sample=images)


class CelebALatentCodecTests(unittest.TestCase):
    def test_from_pretrained_lazily_uses_diffusers_loader(self):
        fake_diffusers = ModuleType("diffusers")

        class FakeAutoencoderKL:
            calls = []

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.calls.append((model_id, kwargs))
                return _MockVAE()

        fake_diffusers.AutoencoderKL = FakeAutoencoderKL
        with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
            codec = CelebALatentCodec.from_pretrained(
                "local/mock-vae",
                local_files_only=True,
                torch_dtype=torch.float16,
            )

        self.assertEqual(codec.model_id, "local/mock-vae")
        self.assertEqual(
            FakeAutoencoderKL.calls,
            [
                (
                    "local/mock-vae",
                    {"local_files_only": True, "torch_dtype": torch.float16},
                )
            ],
        )

    def test_codec_is_frozen_and_stays_in_eval_mode(self):
        vae = _MockVAE()
        codec = CelebALatentCodec(vae)

        self.assertFalse(vae.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in vae.parameters())
        )
        codec.train()
        self.assertTrue(codec.training)
        self.assertFalse(vae.training)

    def test_encode_uses_normalized_posterior_mean_and_config_scaling(self):
        vae = _MockVAE(scaling_factor=0.25)
        codec = CelebALatentCodec(vae)
        images = torch.linspace(0.0, 1.0, 2 * 3 * 64 * 64).reshape(2, 3, 64, 64)

        latents = codec.encode(images)

        normalized = images * 2.0 - 1.0
        pooled = F.avg_pool2d(normalized, kernel_size=8)
        expected_mean = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        self.assertEqual(latents.shape, (2, 4, 8, 8))
        self.assertTrue(torch.allclose(vae.last_encode_input, normalized))
        self.assertTrue(torch.allclose(latents, 0.25 * expected_mean))
        self.assertEqual(vae.last_posterior.sample_calls, 0)
        self.assertFalse(latents.requires_grad)

    def test_posterior_sample_is_fixed_per_image_across_batch_partitions(self):
        codec = CelebALatentCodec(_MockVAE(scaling_factor=0.25))
        images = torch.linspace(0.0, 1.0, 5 * 3 * 64 * 64).reshape(5, 3, 64, 64)

        together = codec.encode_posterior_sample(images, seed=17, index_start=100)
        partitioned = torch.cat(
            (
                codec.encode_posterior_sample(images[:2], seed=17, index_start=100),
                codec.encode_posterior_sample(images[2:], seed=17, index_start=102),
            )
        )
        different_seed = codec.encode_posterior_sample(images, seed=18, index_start=100)
        normalized = images[:1] * 2.0 - 1.0
        pooled = F.avg_pool2d(normalized, kernel_size=8)
        posterior_mean = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        generator = torch.Generator(device="cpu").manual_seed(7456417400580975823)
        expected_first = 0.25 * (
            posterior_mean
            + 0.5
            * torch.randn(
                posterior_mean.shape,
                generator=generator,
                dtype=posterior_mean.dtype,
            )
        )

        self.assertTrue(torch.equal(together, partitioned))
        self.assertTrue(torch.equal(together[:1], expected_first))
        self.assertFalse(torch.equal(together, different_seed))
        self.assertFalse(together.requires_grad)

    def test_posterior_sample_cache_records_and_validates_sampling_contract(self):
        images = torch.rand(5, 3, 64, 64)
        metadata = {"split": "training", "start_index": 100}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path = root / "sample_batch_2.pt"
            second_path = root / "sample_batch_3.pt"
            slice_path = root / "sample_slice.pt"
            first_codec = CelebALatentCodec(_MockVAE(), model_id="mock-vae")
            second_codec = CelebALatentCodec(_MockVAE(), model_id="mock-vae")
            first, first_metadata = encode_loader_to_cache(
                first_codec,
                DataLoader(TensorDataset(images), batch_size=2, shuffle=False),
                first_path,
                metadata=metadata,
                encoding="posterior_sample",
                posterior_sample_seed=29,
                posterior_sample_index_start=100,
            )
            second, _ = encode_loader_to_cache(
                second_codec,
                DataLoader(
                    TensorDataset(images),
                    batch_size=3,
                    shuffle=False,
                    num_workers=2,
                ),
                second_path,
                metadata=metadata,
                encoding="posterior_sample",
                posterior_sample_seed=29,
                posterior_sample_index_start=100,
            )
            sliced, _ = encode_loader_to_cache(
                CelebALatentCodec(_MockVAE(), model_id="mock-vae"),
                DataLoader(TensorDataset(images[2:]), batch_size=2, shuffle=False),
                slice_path,
                metadata={"split": "training", "start_index": 102},
                encoding="posterior_sample",
                posterior_sample_seed=29,
                posterior_sample_index_start=102,
            )

            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(first[2:], sliced))
            self.assertEqual(first_metadata["encoding"], "posterior_sample")
            self.assertEqual(first_metadata["posterior_sample_seed"], 29)
            self.assertEqual(first_metadata["posterior_sample_index_start"], 100)
            self.assertEqual(
                first_metadata["posterior_sample_rng_scheme"],
                POSTERIOR_SAMPLE_RNG_SCHEME,
            )
            self.assertEqual(
                first_metadata["posterior_sample_indexing"],
                POSTERIOR_SAMPLE_INDEXING,
            )
            self.assertIs(first_metadata["posterior_sample_horizontal_flip"], False)
            loaded, _ = load_latent_cache(
                first_path,
                expected_encoding="posterior_sample",
                expected_posterior_sample_seed=29,
                expected_posterior_sample_index_start=100,
            )
            self.assertTrue(torch.equal(loaded, first))

            encode_calls = first_codec.vae.encode_calls
            reused, _ = encode_loader_to_cache(
                first_codec,
                DataLoader(TensorDataset(images), batch_size=1, shuffle=False),
                first_path,
                metadata=metadata,
                encoding="posterior_sample",
                posterior_sample_seed=29,
                posterior_sample_index_start=100,
            )
            self.assertTrue(torch.equal(reused, first))
            self.assertEqual(first_codec.vae.encode_calls, encode_calls)
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                load_latent_cache(
                    first_path,
                    expected_posterior_sample_seed=30,
                )

            malformed_path = root / "malformed_sample.pt"
            malformed = torch.load(first_path, map_location="cpu", weights_only=True)
            malformed["metadata"]["posterior_sample_horizontal_flip"] = True
            torch.save(malformed, malformed_path)
            with self.assertRaisesRegex(ValueError, "must not use horizontal flips"):
                load_latent_cache(malformed_path)

    def test_cache_cli_keeps_mean_default_and_exposes_fixed_sample_mode(self):
        defaults = cache_script.parse_args([])
        sampled = cache_script.parse_args(
            [
                "--encoding",
                "posterior_sample",
                "--posterior-sample-seed",
                "29",
                "--start-index",
                "100",
            ]
        )

        self.assertEqual(defaults.encoding, "posterior_mean")
        self.assertEqual(defaults.posterior_sample_seed, DEFAULT_POSTERIOR_SAMPLE_SEED)
        self.assertEqual(sampled.encoding, "posterior_sample")
        self.assertEqual(sampled.posterior_sample_seed, 29)
        self.assertEqual(sampled.start_index, 100)

    def test_explicit_scaling_override_and_decode_inverse_scaling(self):
        vae = _MockVAE(scaling_factor=0.25)
        codec = CelebALatentCodec(vae, scaling_factor=0.5)
        latents = torch.full((2, 4, 8, 8), 0.125)

        images = codec.decode(latents, clamp=False)

        self.assertTrue(
            torch.allclose(vae.last_decode_input, torch.full_like(latents, 0.25))
        )
        self.assertEqual(images.shape, (2, 3, 64, 64))
        self.assertTrue(torch.allclose(images, torch.full_like(images, 0.625)))

    def test_flatten_round_trip_and_flat_codec_helpers(self):
        codec = CelebALatentCodec(_MockVAE())
        images = torch.rand(3, 3, 64, 64)
        spatial = codec.encode(images)
        flat = flatten_latents(spatial)

        self.assertEqual(flat.shape, (3, CELEBA_LATENT_DIM))
        self.assertTrue(torch.equal(unflatten_latents(flat), spatial))
        self.assertTrue(torch.equal(codec.encode_flat(images), flat))
        self.assertEqual(codec.decode_flat(flat).shape, images.shape)

    def test_cache_round_trip_validates_coordinate_system(self):
        codec = CelebALatentCodec(_MockVAE(), model_id="mock-vae")
        images = torch.rand(5, 3, 64, 64)
        loader = DataLoader(
            TensorDataset(images, torch.arange(images.shape[0])),
            batch_size=2,
            shuffle=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "training_latents.pt"
            latents, metadata = encode_loader_to_cache(
                codec,
                loader,
                path,
                metadata={"split": "training"},
            )
            encode_calls = codec.vae.encode_calls
            reused, reused_metadata = encode_loader_to_cache(codec, loader, path)

            self.assertEqual(latents.shape, (5, 4, 8, 8))
            self.assertEqual(metadata["encoding"], "posterior_mean")
            self.assertFalse(
                any(key.startswith("posterior_sample_") for key in metadata)
            )
            self.assertEqual(metadata["split"], "training")
            self.assertEqual(metadata["model_id"], "mock-vae")
            self.assertTrue(torch.equal(reused, latents))
            self.assertEqual(reused_metadata, metadata)
            self.assertEqual(codec.vae.encode_calls, encode_calls)

            with self.assertRaisesRegex(ValueError, "VAE mismatch"):
                load_latent_cache(path, expected_model_id="different-vae")
            with self.assertRaisesRegex(ValueError, "scaling mismatch"):
                load_latent_cache(path, expected_scaling_factor=0.75)
            with self.assertRaisesRegex(ValueError, "input range mismatch"):
                load_latent_cache(path, expected_input_range="minus_one_one")
            with self.assertRaisesRegex(ValueError, "requested settings.*split"):
                encode_loader_to_cache(
                    codec,
                    loader,
                    path,
                    metadata={"split": "validation"},
                )
            with self.assertRaisesRegex(ValueError, "requested settings.*dtype"):
                encode_loader_to_cache(codec, loader, path, cache_dtype=torch.float16)
            with self.assertRaisesRegex(ValueError, "requested settings.*max_items"):
                encode_loader_to_cache(codec, loader, path, max_items=4)

    def test_local_vae_file_hashes_are_stored_and_rechecked_before_cache_save(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "vae"
            model_dir.mkdir()
            weights = model_dir / "weights.bin"
            weights.write_bytes(b"first weights")
            expected_hashes = directory_file_hashes(model_dir)
            codec = CelebALatentCodec(
                _MockVAE(),
                model_id=str(model_dir),
                model_files_sha256=expected_hashes,
            )
            latents = torch.zeros(2, 4, 8, 8)
            cache = Path(temporary_directory) / "cache.pt"

            save_latent_cache(cache, latents, codec=codec)
            _, metadata = load_latent_cache(cache)
            self.assertEqual(metadata["vae_files_sha256"], expected_hashes)

            weights.write_bytes(b"changed weights")
            with self.assertRaisesRegex(RuntimeError, "VAE files changed"):
                save_latent_cache(cache, latents, codec=codec)


class LatentGaussianPriorTests(unittest.TestCase):
    def test_train_only_full_fit_matches_population_moments_and_round_trips(self):
        values = _well_conditioned_latents()
        prior = LatentGaussianPrior.fit_full(
            values,
            train_start_index=17,
            train_end_index=17 + values.shape[0],
            source_cache_sha256="a" * 64,
        )
        restored = LatentGaussianPrior.from_payload(prior.to_payload())

        self.assertTrue(prior.same_definition(restored))
        self.assertTrue(torch.allclose(prior.mean, torch.full_like(prior.mean, 0.25)))
        self.assertTrue(
            torch.allclose(
                prior.covariance, torch.eye(CELEBA_LATENT_DIM, dtype=torch.float64)
            )
        )
        metadata = prior.metadata()
        self.assertEqual(metadata["fit_start_inclusive"], 17)
        self.assertEqual(metadata["fit_stop_exclusive"], 17 + values.shape[0])
        self.assertEqual(metadata["estimator"], "population_moments_correction_0")

    def test_full_sampler_is_exactly_invariant_to_outer_batch_partition(self):
        values = _well_conditioned_latents(mean=-0.4)
        prior = LatentGaussianPrior.fit_full(
            values,
            train_start_index=0,
            train_end_index=values.shape[0],
            source_cache_sha256="b" * 64,
        )
        sampler = prior.sampler(device="cpu", dtype=torch.float64)
        noise = torch.randn(
            7, CELEBA_LATENT_DIM, generator=torch.Generator().manual_seed(9)
        )

        together = sampler.transform_standard(noise, rowwise=True)
        partitioned = torch.cat(
            [
                sampler.transform_standard(noise[:2], rowwise=True),
                sampler.transform_standard(noise[2:6], rowwise=True),
                sampler.transform_standard(noise[6:], rowwise=True),
            ]
        )

        self.assertTrue(torch.equal(together, partitioned))
        self.assertEqual(together.dtype, noise.dtype)

    def test_sidecar_rejects_tensor_corruption(self):
        values = _well_conditioned_latents()
        prior = LatentGaussianPrior.fit_full(
            values,
            train_start_index=0,
            train_end_index=values.shape[0],
            source_cache_sha256="c" * 64,
        )
        payload = prior.to_payload()
        payload["mean"] = payload["mean"].clone()
        payload["mean"][0] += 1.0

        with self.assertRaisesRegex(ValueError, "metadata or tensor hashes"):
            LatentGaussianPrior.from_payload(payload)

    def test_sidecar_tolerates_only_last_bit_changes_in_derived_spectral_metadata(self):
        prior = LatentGaussianPrior.fit_full(
            _well_conditioned_latents(),
            train_start_index=0,
            train_end_index=2 * CELEBA_LATENT_DIM,
            source_cache_sha256="e" * 64,
        )
        payload = prior.to_payload()
        payload["metadata"] = dict(payload["metadata"])
        payload["metadata"]["eigenvalue_min"] += 5e-15

        restored = LatentGaussianPrior.from_payload(payload)
        self.assertTrue(restored.same_definition(prior))

        payload["metadata"]["eigenvalue_min"] += 1e-3
        with self.assertRaisesRegex(ValueError, "metadata or tensor hashes"):
            LatentGaussianPrior.from_payload(payload)

    def test_full_fit_rejects_singular_covariance_without_hidden_jitter(self):
        values = torch.zeros(2 * CELEBA_LATENT_DIM, CELEBA_LATENT_DIM)

        with self.assertRaisesRegex(ValueError, "does not add jitter or shrinkage"):
            LatentGaussianPrior.fit_full(
                values,
                train_start_index=0,
                train_end_index=values.shape[0],
                source_cache_sha256="d" * 64,
            )

    def test_standard_prior_transform_is_the_legacy_identity(self):
        values = torch.randn(3, CELEBA_LATENT_DIM)
        transformed = (
            LatentGaussianPrior.standard().sampler().transform_standard(values)
        )

        self.assertIs(transformed, values)


class TransitionConditionedLatentUNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TransitionConditionedLatentUNet(
            base_channels=8,
            time_embedding_dim=8,
            condition_dim=16,
        )

    def test_diagonal_is_exact_identity_for_scalar_and_batched_times(self):
        x_flat = torch.randn(3, CELEBA_LATENT_DIM)
        scalar_output = self.model(x_flat, 0.37, 0.37)
        times = torch.tensor([0.0, 0.4, 1.0])
        batched_output = self.model(x_flat, times, times)

        self.assertTrue(torch.equal(scalar_output, x_flat))
        self.assertTrue(torch.equal(batched_output, x_flat))

    def test_model_accepts_spatial_and_flattened_latents(self):
        x_spatial = torch.randn(2, 4, 8, 8)
        output_spatial = self.model(x_spatial, torch.zeros(2), torch.full((2,), 0.5))
        output_flat = self.model(flatten_latents(x_spatial), 0.0, 0.5)

        self.assertEqual(output_spatial.shape, x_spatial.shape)
        self.assertEqual(output_flat.shape, (2, CELEBA_LATENT_DIM))
        self.assertTrue(torch.allclose(flatten_latents(output_spatial), output_flat))

    def test_transition_to_one_suppresses_explicit_source(self):
        # Make g_theta a known constant independent of x.  No sigmoid/tanh is
        # present: a value greater than one must remain greater than one.
        with torch.no_grad():
            self.model.output_conv.weight.zero_()
            self.model.output_conv.bias.fill_(2.5)
        x1 = torch.randn(2, CELEBA_LATENT_DIM)
        x2 = torch.randn(2, CELEBA_LATENT_DIM) + 20.0
        start = torch.tensor([0.0, 0.83])

        output1 = self.model(x1, start, torch.ones(2))
        output2 = self.model(x2, start, torch.ones(2))

        expected = torch.full_like(x1, 2.5)
        self.assertTrue(torch.equal(output1, expected))
        self.assertTrue(torch.equal(output2, expected))
        self.assertGreater(float(output1.detach().max()), 1.0)

    def test_intermediate_map_uses_mnist_endpoint_interpolation(self):
        with torch.no_grad():
            self.model.output_conv.weight.zero_()
            self.model.output_conv.bias.fill_(-0.75)
        x = torch.randn(2, 4, 8, 8)
        start = torch.tensor([0.0, 0.25])
        end = torch.tensor([0.5, 0.625])
        weight = flow_map_endpoint_weight(
            start,
            end,
            batch_size=2,
            device=x.device,
            dtype=x.dtype,
        )

        output = self.model(x, start, end)
        expected = (1.0 - weight[:, :, None, None]) * x + weight[
            :, :, None, None
        ] * -0.75

        self.assertTrue(torch.allclose(output, expected))

    def test_retained_source_coefficients_telescope(self):
        times = torch.tensor([0.12, 0.31, 0.64, 0.91])
        retention = torch.ones(1, 1)
        for start, end in zip(times[:-1], times[1:]):
            weight = flow_map_endpoint_weight(
                start,
                end,
                batch_size=1,
                device=times.device,
                dtype=times.dtype,
            )
            retention = retention * (1.0 - weight)
        expected = (1.0 - times[-1]) / (1.0 - times[0])

        self.assertTrue(torch.allclose(retention.squeeze(), expected, atol=1e-7))

        terminal_weight = flow_map_endpoint_weight(
            times[-1],
            1.0,
            batch_size=1,
            device=times.device,
            dtype=times.dtype,
        )
        self.assertTrue(
            torch.equal(
                retention * (1.0 - terminal_weight), torch.zeros_like(retention)
            )
        )

    def test_endpoint_derivative_exists_on_diagonal_for_wgf_jvp(self):
        # The endpoint loss differentiates Phi with respect to s at s=t.  The
        # exact identity must not be implemented through a detached shortcut.
        with torch.no_grad():
            self.model.output_conv.weight.zero_()
            self.model.output_conv.bias.fill_(0.5)
        x = torch.zeros(1, CELEBA_LATENT_DIM)
        start = torch.tensor(0.25)
        end = torch.tensor(0.25, requires_grad=True)

        output = self.model(x, start, end)
        derivative = torch.autograd.grad(output.sum(), end)[0]

        self.assertTrue(torch.equal(output, x))
        self.assertGreater(float(derivative), 0.0)

    def test_residual_zero_init_is_exact_identity_without_changing_state_dict(self):
        residual = TransitionConditionedLatentUNet(
            base_channels=8,
            time_embedding_dim=8,
            condition_dim=16,
            endpoint_parameterization="residual",
        )
        absolute = TransitionConditionedLatentUNet(
            base_channels=8,
            time_embedding_dim=8,
            condition_dim=16,
            endpoint_parameterization="absolute",
        )
        x = torch.randn(3, CELEBA_LATENT_DIM)
        starts = torch.tensor([0.0, 0.2, 0.8])
        ends = torch.tensor([1.0, 0.7, 0.9])

        self.assertTrue(torch.equal(residual(x, starts, ends), x))
        self.assertTrue(torch.equal(residual.endpoint(x, starts, ends), x))
        self.assertTrue(torch.equal(absolute(x, 0.0, 1.0), torch.zeros_like(x)))
        self.assertEqual(set(residual.state_dict()), set(absolute.state_dict()))


if __name__ == "__main__":
    unittest.main()
