import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flowmaps.objectives.wgf as wgf_module
from flowmaps.models.toy_mnist import (
    CELEBA_DIM,
    CelebaUNetGenerator,
    TimeConditionedMLP,
    TimeConditionedMnistConvGenerator,
    TransitionConditionedCelebaUNet,
    TransitionConditionedMLP,
    TransitionConditionedMnistConvGenerator,
    prepare_time_features,
)
from flowmaps.objectives.wgf import (
    assemble_global_support,
    beta_value,
    euler_flow_matching_loss,
    evaluate_flow_map,
    fast_wgf_flow_map_loss,
    gather_global_support,
    kde_score,
    kl_kde_wgf_velocity,
    mmd_wgf_velocity,
    optimizer_time_drifting_loss,
    sample_wgf_flow_map,
    schedule_delta,
    transition_delta,
    wgf_flow_map_loss,
    wgf_time,
    wgf_velocity_diagnostics,
)


class IdentityGenerator(torch.nn.Module):
    def forward(self, z):
        return z


class IdentityFlow(torch.nn.Module):
    def forward(self, z, t):
        _ = t
        return z


class ShiftFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor([0.1, -0.2]))

    def forward(self, z, t):
        return z + t.reshape(-1, 1) * self.shift


class TerminalCountingFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.zeros(2))
        self.terminal_calls = 0

    def forward(self, z, t):
        if torch.all(t == 1):
            self.terminal_calls += 1
        return z + t.reshape(-1, 1) * self.shift


class IdentityFlowMap(torch.nn.Module):
    def forward(self, x, t, s):
        _ = t, s
        return x


class ConstantFlowMap(torch.nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, x, t, s):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1, 1)
        s = torch.as_tensor(s, device=x.device, dtype=x.dtype).reshape(-1, 1)
        return x + (s - t) * self.shift.expand_as(x)


class StartDependentFlowMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor([0.1, -0.2]))

    def forward(self, x, t, s):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1, 1)
        s = torch.as_tensor(s, device=x.device, dtype=x.dtype).reshape(-1, 1)
        return x + (s - t) * (1.0 + t) * self.shift.expand_as(x)


class WgfFlowTests(unittest.TestCase):
    def test_prepare_time_features_broadcasts_scalar(self):
        times = prepare_time_features(
            0.25,
            batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(times.shape, (4, 1))
        self.assertTrue(torch.allclose(times, 0.25 * torch.ones(4, 1)))

    def test_time_conditioned_mlp_is_identity_at_zero_time(self):
        model = TimeConditionedMLP(in_dim=2, hidden=16, out_dim=2)
        z = torch.randn(8, 2)
        self.assertTrue(torch.allclose(model(z, torch.zeros(8)), z))

    def test_time_conditioned_mnist_conv_generator_output_shape(self):
        model = TimeConditionedMnistConvGenerator(base_channels=8)
        z = torch.randn(4, 28 * 28)
        out0 = model(z, torch.zeros(4))
        out1 = model(z, torch.ones(4))
        self.assertEqual(out1.shape, (4, 28 * 28))
        self.assertTrue(torch.allclose(out0, z))
        self.assertTrue(torch.all((0.0 <= out1) & (out1 <= 1.0)))

    def test_transition_conditioned_mlp_output_shape(self):
        model = TransitionConditionedMLP(in_dim=2, hidden=16, out_dim=2)
        x = torch.randn(8, 2)
        out = model(x, torch.zeros(8), torch.ones(8))
        self.assertEqual(out.shape, (8, 2))
        self.assertTrue(torch.allclose(model(x, torch.zeros(8), torch.zeros(8)), x))

    def test_transition_conditioned_mnist_conv_generator_output_shape(self):
        model = TransitionConditionedMnistConvGenerator(base_channels=8)
        x = torch.randn(4, 28 * 28)
        out = model(x, torch.zeros(4), torch.ones(4))
        out_from_midpoint = model(
            4.0 * torch.ones_like(x), 0.5 * torch.ones(4), torch.ones(4)
        )
        self.assertEqual(out.shape, (4, 28 * 28))
        self.assertTrue(torch.all((0.0 <= out) & (out <= 1.0)))
        self.assertTrue(
            torch.all((0.0 <= out_from_midpoint) & (out_from_midpoint <= 1.0))
        )
        self.assertTrue(torch.allclose(model(x, torch.zeros(4), torch.zeros(4)), x))

    def test_transition_conditioned_celeba_unet_output_shape(self):
        model = TransitionConditionedCelebaUNet(base_channels=8)
        x = torch.randn(2, CELEBA_DIM)
        out = model(x, torch.zeros(2), torch.ones(2))
        self.assertEqual(out.shape, (2, CELEBA_DIM))
        self.assertTrue(torch.all((0.0 <= out) & (out <= 1.0)))
        self.assertTrue(torch.allclose(model(x, torch.zeros(2), torch.zeros(2)), x))

    def test_celeba_unet_generator_output_shape(self):
        model = CelebaUNetGenerator(base_channels=8)
        out = model(torch.randn(2, CELEBA_DIM))
        self.assertEqual(out.shape, (2, CELEBA_DIM))
        self.assertTrue(torch.all((0.0 <= out) & (out <= 1.0)))

    def test_constant_schedule_delta(self):
        delta = schedule_delta(
            torch.tensor(0.2), 0.1, schedule="constant", time_scale=3.0
        )
        self.assertAlmostEqual(float(delta), 0.3, places=6)

    def test_rational_schedule_matches_requested_time_map(self):
        t = torch.tensor(0.25)
        eps = 0.01
        time_scale = 2.0
        expected_s = time_scale * t / (1.0 - t + eps)
        expected_beta = time_scale * (1.0 + eps) / (1.0 - t + eps) ** 2

        self.assertAlmostEqual(
            float(wgf_time(t, schedule="rational", time_scale=time_scale, eps=eps)),
            float(expected_s),
            places=6,
        )
        self.assertAlmostEqual(
            float(beta_value(t, schedule="rational", time_scale=time_scale, eps=eps)),
            float(expected_beta),
            places=6,
        )

    def test_mmd_velocity_vanishes_when_batches_match(self):
        x = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        velocity = mmd_wgf_velocity(x, x, x, bandwidth=0.5)
        self.assertTrue(torch.allclose(velocity, torch.zeros_like(velocity), atol=1e-6))

    def test_kl_kde_velocity_vanishes_when_batches_match(self):
        x = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        velocity = kl_kde_wgf_velocity(x, x, x, bandwidth=0.5)
        self.assertTrue(torch.allclose(velocity, torch.zeros_like(velocity), atol=1e-6))

    def test_kl_kde_velocity_is_score_difference(self):
        x = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        data = torch.tensor([[0.2, 0.1], [1.5, -0.7]], dtype=torch.float32)
        reference = torch.tensor([[-0.3, 0.4], [0.7, -1.2]], dtype=torch.float32)
        velocity = kl_kde_wgf_velocity(x, data, reference, bandwidth=0.75)
        expected = kde_score(x, data, bandwidth=0.75) - kde_score(
            x, reference, bandwidth=0.75
        )
        self.assertTrue(torch.allclose(velocity, expected))

    def test_kde_score_same_batch_excludes_self_term(self):
        x = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
        score = kde_score(x, x, bandwidth=1.0, same_batch=True)
        expected = torch.tensor([[2.0], [-2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(score, expected))

    def test_mmd_velocity_same_batch_excludes_reference_diagonal(self):
        x = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
        data = torch.zeros_like(x)
        velocity = mmd_wgf_velocity(x, data, x, bandwidth=1.0, same_batch=True)

        data_diff = data.unsqueeze(0) - x.unsqueeze(1)
        ref_diff = x.unsqueeze(0) - x.unsqueeze(1)
        k_data = torch.exp(-data_diff.square().sum(dim=-1) / 2.0)
        k_ref = torch.exp(-ref_diff.square().sum(dim=-1) / 2.0)
        data_term = (data_diff * k_data.unsqueeze(-1)).mean(dim=1)
        mask = ~torch.eye(x.shape[0], dtype=torch.bool)
        ref_term = (
            (ref_diff * k_ref.unsqueeze(-1))[mask]
            .view(x.shape[0], x.shape[0] - 1, -1)
            .mean(dim=1)
        )
        expected = data_term - ref_term

        self.assertTrue(torch.allclose(velocity, expected))

    def test_dimension_normalized_mmd_preserves_per_coordinate_velocity_scale(self):
        one_dimensional_x = torch.zeros(1, 1)
        one_dimensional_data = torch.ones(1, 1)
        one_dimensional_reference = torch.zeros(1, 1)
        expected = mmd_wgf_velocity(
            one_dimensional_x,
            one_dimensional_data,
            one_dimensional_reference,
            bandwidth=1.0,
            rbf_geometry="dimension_normalized",
        )[0, 0]

        dimension = 12_288
        velocity = mmd_wgf_velocity(
            one_dimensional_x.expand(1, dimension),
            one_dimensional_data.expand(1, dimension),
            one_dimensional_reference.expand(1, dimension),
            bandwidth=1.0,
            rbf_geometry="dimension_normalized",
        )

        self.assertTrue(
            torch.allclose(velocity, expected.expand_as(velocity), atol=1e-6)
        )

    def test_dimension_normalized_kl_preserves_per_coordinate_velocity_scale(self):
        x = torch.zeros(1, 1)
        data = torch.tensor([[1.0], [2.0]])
        reference = torch.tensor([[-1.0], [-2.0]])
        expected = kl_kde_wgf_velocity(
            x,
            data,
            reference,
            bandwidth=1.0,
            rbf_geometry="dimension_normalized",
        )[0, 0]

        dimension = 128
        velocity = kl_kde_wgf_velocity(
            x.expand(1, dimension),
            data.expand(2, dimension),
            reference.expand(2, dimension),
            bandwidth=1.0,
            rbf_geometry="dimension_normalized",
        )

        self.assertTrue(
            torch.allclose(velocity, expected.expand_as(velocity), atol=1e-6)
        )

    def test_dimension_normalized_geometry_prevents_celeba_mmd_underflow(self):
        dimension = 64 * 64 * 3
        x = torch.zeros(2, dimension)
        data = torch.ones(3, dimension)
        reference = -torch.ones(3, dimension)

        raw = mmd_wgf_velocity(x, data, reference, bandwidth=1.0)
        normalized = mmd_wgf_velocity(
            x,
            data,
            reference,
            bandwidth=1.0,
            rbf_geometry="dimension_normalized",
        )
        raw_diagnostics = wgf_velocity_diagnostics(
            "mmd",
            x,
            data,
            reference,
            bandwidth=1.0,
        )
        normalized_diagnostics = wgf_velocity_diagnostics(
            "mmd",
            x,
            data,
            reference,
            bandwidth=1.0,
            rbf_geometry="dimension_normalized",
        )

        self.assertEqual(int(torch.count_nonzero(raw)), 0)
        self.assertEqual(float(raw_diagnostics["attraction_kernel_mean"]), 0.0)
        self.assertEqual(float(raw_diagnostics["attraction_ess"]), 0.0)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertGreater(float(normalized.abs().min()), 0.0)
        self.assertGreater(float(normalized_diagnostics["attraction_kernel_mean"]), 0.0)
        self.assertGreater(float(normalized_diagnostics["attraction_ess"]), 1.0)
        self.assertGreater(float(normalized_diagnostics["attraction_norm"]), 0.0)
        self.assertGreater(float(normalized_diagnostics["repulsion_norm"]), 0.0)

    def test_global_support_assembly_keeps_time_groups_contiguous(self):
        rank_zero = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
        rank_one = torch.tensor([[2.0], [12.0]])

        support = assemble_global_support(
            [rank_zero, rank_one],
            time_groups=2,
        )

        expected = torch.tensor([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
        self.assertTrue(torch.equal(support, expected))

    def test_distributed_helpers_are_noops_when_distributed_is_unavailable(self):
        support = torch.randn(3, 2)
        value = torch.tensor([0.25, 0.75])
        with mock.patch.object(wgf_module, "distributed_is_active", return_value=False):
            self.assertIs(gather_global_support(support, time_groups=1), support)
            self.assertIs(
                wgf_module.synchronize_distributed_tensor(value, enabled=True),
                value,
            )

    def test_euler_loss_is_zero_for_identity_flow_with_zero_velocity(self):
        model = IdentityFlow()
        z = torch.tensor([[0.2, -0.1], [1.0, 0.5]], dtype=torch.float32)
        loss, stats = euler_flow_matching_loss(
            model,
            z,
            z,
            z,
            dt=0.1,
            bandwidth=0.75,
            lambda_terminal=1.0,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_dyn"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_terminal"]), 0.0, places=6)

    def test_evaluate_flow_map_stacks_time_axis(self):
        model = IdentityFlow()
        z = torch.randn(5, 2)
        outputs = evaluate_flow_map(model, z, [0.0, 0.5, 1.0])
        self.assertEqual(outputs.shape, (5, 3, 2))
        self.assertTrue(torch.allclose(outputs[:, 0], z))
        self.assertTrue(torch.allclose(outputs[:, 2], z))

    def test_transition_delta_matches_linear_schedule(self):
        delta = transition_delta(torch.tensor(0.2), torch.tensor(0.3), time_scale=3.0)
        self.assertAlmostEqual(float(delta), 0.3, places=6)

    def test_sample_wgf_flow_map_composes_transitions(self):
        model = ConstantFlowMap([1.0, -0.5])
        z = torch.zeros(2, 2)
        samples = sample_wgf_flow_map(model, z, num_steps=2)
        expected = torch.tensor([[1.0, -0.5], [1.0, -0.5]])
        self.assertTrue(torch.allclose(samples, expected))

    def test_optimizer_time_drifting_loss_is_zero_when_velocity_is_zero(self):
        model = IdentityGenerator()
        z = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32)
        loss, stats = optimizer_time_drifting_loss(
            model,
            z,
            z,
            z,
            step_size=0.5,
            bandwidth=0.6,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["velocity_norm"]), 0.0, places=6)

    def test_optimizer_time_drifting_supports_global_normalized_kde(self):
        model = IdentityGenerator()
        z = torch.zeros(3, 2)
        gather_calls = []

        def recording_gather(support, *, time_groups=1):
            gather_calls.append((support.shape, time_groups))
            return support

        with mock.patch.object(
            wgf_module, "gather_global_support", side_effect=recording_gather
        ):
            loss, stats = optimizer_time_drifting_loss(
                model,
                z,
                z,
                torch.ones_like(z),
                step_size=0.1,
                bandwidth=1.0,
                rbf_geometry="dimension_normalized",
                global_kde_support=True,
            )

        self.assertEqual(gather_calls, [((3, 2), 1), ((3, 2), 1)])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(stats["velocity_norm"]), 0.0)

    def test_terminal_loss_contributes_gradient(self):
        z = torch.zeros(4, 2)
        z_ref = torch.zeros(4, 2)
        data = torch.ones(4, 2)

        model = ShiftFlow()
        torch.manual_seed(0)
        loss, _ = euler_flow_matching_loss(
            model,
            z,
            z_ref,
            data,
            dt=0.1,
            lambda_terminal=0.0,
            bandwidth=1.0,
        )
        loss.backward()
        grad_without_terminal = model.shift.grad.detach().clone()

        model.zero_grad()
        torch.manual_seed(0)
        loss, _ = euler_flow_matching_loss(
            model,
            z,
            z_ref,
            data,
            dt=0.1,
            lambda_terminal=1.0,
            bandwidth=1.0,
        )
        loss.backward()
        grad_with_terminal = model.shift.grad.detach().clone()

        self.assertFalse(torch.allclose(grad_without_terminal, grad_with_terminal))

    def test_terminal_loss_can_skip_endpoint_branch(self):
        model = TerminalCountingFlow()
        z = torch.zeros(4, 2)
        loss, stats = euler_flow_matching_loss(
            model,
            z,
            z,
            z,
            dt=0.1,
            lambda_terminal=0.0,
            bandwidth=1.0,
        )
        self.assertEqual(model.terminal_calls, 0)
        self.assertAlmostEqual(float(stats["loss_terminal"]), 0.0, places=6)
        self.assertTrue(torch.isfinite(loss))

    def test_wgf_flow_map_loss_is_zero_when_velocity_is_zero(self):
        model = IdentityFlowMap()
        z = torch.tensor([[0.2, -0.1], [1.0, 0.5]], dtype=torch.float32)
        loss, stats = wgf_flow_map_loss(
            model,
            z,
            z,
            z,
            eps_t=0.25,
            time_scale=1.0,
            lambda_sg=1.0,
            bandwidth=0.75,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_wgf"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_sg"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["semigroup_error"]), 0.0, places=6)

    def test_wgf_flow_map_loss_contributes_gradient(self):
        z = torch.zeros(4, 2)
        z_ref = torch.zeros(4, 2)
        data = torch.ones(4, 2)

        model = ConstantFlowMap([0.1, -0.2])
        torch.manual_seed(0)
        loss, _ = wgf_flow_map_loss(
            model,
            z,
            z_ref,
            data,
            eps_t=0.25,
            time_scale=1.0,
            lambda_sg=0.0,
            bandwidth=1.0,
        )
        loss.backward()
        self.assertGreater(float(model.shift.grad.norm()), 0.0)

    def test_wgf_flow_map_loss_uses_eps_t(self):
        model = IdentityFlowMap()
        z = torch.zeros(2, 2)
        _, stats = wgf_flow_map_loss(
            model,
            z,
            z,
            z,
            eps_t=0.2,
            time_scale=3.0,
            lambda_sg=0.0,
            bandwidth=1.0,
        )
        self.assertAlmostEqual(float(stats["delta_t"]), 0.6, places=6)

    def test_wgf_flow_map_loss_matches_forward_local_transition(self):
        shift_values = [0.3, -0.1]
        shift = torch.tensor(shift_values)
        model = ConstantFlowMap(shift_values)
        z = torch.randn(4, 2)

        original_velocity = wgf_module.wgf_velocity

        def constant_velocity(kind, x, data, reference, bandwidth):
            _ = kind, data, reference, bandwidth
            return shift.to(device=x.device, dtype=x.dtype).expand_as(x)

        try:
            wgf_module.wgf_velocity = constant_velocity
            loss, stats = wgf_flow_map_loss(
                model,
                z,
                z,
                z,
                eps_t=0.25,
                time_scale=1.0,
                lambda_sg=1.0,
            )
        finally:
            wgf_module.wgf_velocity = original_velocity

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_wgf"]), 0.0, places=6)

    def test_wgf_flow_map_loss_supports_time_groups(self):
        model = TransitionConditionedMLP(in_dim=2, hidden=16, out_dim=2)
        z = torch.randn(4, 2)
        loss, stats = wgf_flow_map_loss(
            model,
            z,
            torch.randn_like(z),
            torch.randn_like(z),
            dt=0.1,
            bandwidth=1.0,
            time_groups=2,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(stats["path_time_std"]), 0.0)

    def test_wgf_flow_map_jvp_teacher_is_explicitly_detached(self):
        model = ConstantFlowMap([0.1, -0.2])
        teacher_probe = torch.nn.Parameter(torch.tensor(1.0))
        z = torch.zeros(4, 2)
        derivative_calls = 0

        def controlled_endpoint_derivative(model_arg, x, t, s):
            nonlocal derivative_calls
            derivative_calls += 1
            primal = model_arg(x, t, s)
            tangent = torch.zeros_like(primal)
            if derivative_calls == 3:
                # Mirror autograd.functional.jvp(create_graph=True), which can
                # build a graph even when its caller is inside torch.no_grad().
                with torch.enable_grad():
                    tangent = teacher_probe * torch.ones_like(primal)
            return primal, tangent

        with mock.patch.object(
            wgf_module,
            "endpoint_time_derivative",
            side_effect=controlled_endpoint_derivative,
        ):
            loss, _ = wgf_flow_map_loss(
                model,
                z,
                z,
                torch.ones_like(z),
                dt=0.1,
                lambda_sg=0.0,
                bandwidth=1.0,
            )
        loss.backward()

        self.assertEqual(derivative_calls, 3)
        self.assertIsNone(teacher_probe.grad)

    def test_fast_wgf_flow_map_loss_is_zero_for_exact_map(self):
        shift = torch.tensor([0.3, -0.1])
        model = ConstantFlowMap([0.3, -0.1])
        z = torch.randn(4, 2)
        original_velocity = wgf_module.wgf_velocity

        def constant_velocity(kind, x, data, reference, bandwidth):
            _ = kind, data, reference, bandwidth
            return shift.to(device=x.device, dtype=x.dtype).expand_as(x)

        try:
            wgf_module.wgf_velocity = constant_velocity
            loss, stats = fast_wgf_flow_map_loss(
                model,
                z,
                z,
                z,
                dt=0.25,
                time_scale=1.0,
                path_weight=1.0,
                lambda_eulerian=1.0,
                lambda_sg=1.0,
            )
        finally:
            wgf_module.wgf_velocity = original_velocity

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_wgf"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_lagrangian"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_eulerian"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["loss_sg"]), 0.0, places=6)

    def test_fast_wgf_flow_map_loss_contributes_gradient(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)
        loss, stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            torch.ones_like(z),
            dt=0.25,
            time_scale=1.0,
            bandwidth=1.0,
        )
        loss.backward()
        self.assertGreater(float(model.shift.grad.norm()), 0.0)
        self.assertNotIn("loss_sg", stats)
        self.assertIn("velocity_rms", stats)
        self.assertIn("kde_attraction_ess", stats)
        self.assertIn("kde_repulsion_ess", stats)

    def test_fast_loss_gathers_grouped_and_boundary_kde_supports(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)
        gather_calls = []

        def recording_gather(support, *, time_groups=1):
            gather_calls.append((support.shape[0], time_groups))
            return support

        with mock.patch.object(
            wgf_module, "gather_global_support", side_effect=recording_gather
        ):
            fast_wgf_flow_map_loss(
                model,
                z,
                z,
                torch.ones_like(z),
                dt=0.1,
                lambda_boundary=1.0,
                time_groups=2,
                bandwidth=1.0,
                global_kde_support=True,
            )

        self.assertEqual(gather_calls, [(4, 1), (4, 2), (4, 1), (4, 1)])

    def test_fast_loss_propagates_normalized_geometry_to_all_wgf_terms(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)
        geometries = []

        def recording_velocity(
            kind,
            x,
            data,
            reference,
            bandwidth,
            *,
            rbf_geometry,
        ):
            _ = kind, data, reference, bandwidth
            geometries.append(rbf_geometry)
            return torch.ones_like(x)

        with mock.patch.object(
            wgf_module, "wgf_velocity", side_effect=recording_velocity
        ):
            fast_wgf_flow_map_loss(
                model,
                z,
                z,
                z,
                dt=0.1,
                lambda_boundary=1.0,
                time_groups=2,
                rbf_geometry="dimension_normalized",
            )

        self.assertEqual(geometries, ["dimension_normalized"] * 4)

    def test_fast_loss_synchronizes_all_sampled_times_when_requested(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)
        synchronized_shapes = []

        def recording_synchronization(value, *, enabled):
            self.assertTrue(enabled)
            synchronized_shapes.append(tuple(value.shape))
            return value

        with mock.patch.object(
            wgf_module,
            "synchronize_distributed_tensor",
            side_effect=recording_synchronization,
        ):
            fast_wgf_flow_map_loss(
                model,
                z,
                z,
                torch.ones_like(z),
                dt=0.1,
                lambda_boundary=1.0,
                time_groups=2,
                bandwidth=1.0,
                synchronize_times=True,
            )

        self.assertEqual(synchronized_shapes, [(2,), (4,), ()])

    def test_fast_boundary_step_size_scales_boundary_euler_target(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)

        torch.manual_seed(0)
        _, full_stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            torch.ones_like(z),
            dt=0.1,
            path_weight=0.0,
            lambda_boundary_origin=1.0,
            lambda_boundary_arbitrary=0.0,
            boundary_step_size=1.0,
            bandwidth=1.0,
        )
        torch.manual_seed(0)
        _, half_stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            torch.ones_like(z),
            dt=0.1,
            path_weight=0.0,
            lambda_boundary_origin=1.0,
            lambda_boundary_arbitrary=0.0,
            boundary_step_size=0.5,
            bandwidth=1.0,
        )

        self.assertTrue(
            torch.allclose(
                half_stats["loss_boundary_origin"],
                0.25 * full_stats["loss_boundary_origin"],
            )
        )
        self.assertAlmostEqual(float(half_stats["boundary_step_size"]), 0.5)

    def test_fast_boundary_loss_contributes_gradient(self):
        z = torch.zeros(4, 2)
        data = torch.ones_like(z)
        model = ConstantFlowMap([0.1, -0.2])

        torch.manual_seed(0)
        loss, _ = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            data,
            dt=0.25,
            time_scale=1.0,
            lambda_boundary=0.0,
            bandwidth=1.0,
        )
        loss.backward()
        grad_without_boundary = model.shift.grad.detach().clone()

        model.zero_grad()
        torch.manual_seed(0)
        loss, stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            data,
            dt=0.25,
            time_scale=1.0,
            lambda_boundary=1.0,
            bandwidth=1.0,
        )
        loss.backward()

        self.assertFalse(torch.allclose(grad_without_boundary, model.shift.grad))
        self.assertGreater(float(stats["loss_boundary_origin"]), 0.0)
        self.assertGreater(float(stats["loss_boundary_arbitrary"]), 0.0)

    def test_fast_wgf_term_can_be_disabled_independently(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)
        loss, stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            torch.ones_like(z),
            dt=0.25,
            time_scale=1.0,
            path_weight=1.0,
            lambda_wgf=0.0,
            lambda_lagrangian=0.0,
            lambda_eulerian=0.0,
            lambda_boundary=0.0,
            bandwidth=1.0,
        )

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        self.assertGreater(float(stats["loss_wgf"]), 0.0)

    def test_fast_boundary_terms_have_independent_weights(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)
        loss, stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            torch.ones_like(z),
            dt=0.25,
            time_scale=1.0,
            path_weight=0.0,
            lambda_boundary=0.0,
            lambda_boundary_origin=0.0,
            lambda_boundary_arbitrary=1.0,
            bandwidth=1.0,
        )

        self.assertAlmostEqual(
            float(loss.detach()),
            float(stats["loss_boundary_arbitrary"]),
            places=7,
        )
        self.assertAlmostEqual(float(stats["lambda_boundary_origin"]), 0.0, places=7)
        self.assertAlmostEqual(float(stats["lambda_boundary_arbitrary"]), 1.0, places=7)

    def test_fast_terminal_semigroup_targets_the_direct_endpoint(self):
        model = StartDependentFlowMap()
        z = torch.zeros(4, 2)
        loss, stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            torch.ones_like(z),
            dt=0.1,
            path_weight=0.0,
            lambda_sg=0.0,
            lambda_terminal_sg=1.0,
            lambda_boundary_origin=0.0,
            lambda_boundary_arbitrary=0.0,
            bandwidth=1.0,
        )

        self.assertGreater(float(stats["loss_terminal_semigroup"]), 0.0)
        self.assertAlmostEqual(
            float(loss.detach()),
            float(stats["loss_terminal_semigroup"]),
            places=7,
        )
        loss.backward()
        self.assertGreater(float(model.shift.grad.norm()), 0.0)

    def test_fast_terminal_semigroup_vanishes_for_exact_flow_map(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.randn(4, 2)
        loss, stats = fast_wgf_flow_map_loss(
            model,
            z,
            z,
            z,
            dt=0.1,
            path_weight=0.0,
            lambda_sg=0.0,
            lambda_terminal_sg=1.0,
            lambda_boundary_origin=0.0,
            lambda_boundary_arbitrary=0.0,
            bandwidth=1.0,
        )

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        self.assertAlmostEqual(float(stats["loss_terminal_semigroup"]), 0.0, places=7)

    def test_fast_arbitrary_boundary_loss_is_independent(self):
        model = StartDependentFlowMap()
        z = torch.zeros(4, 2)
        original_velocity = wgf_module.wgf_velocity
        calls = 0

        def arbitrary_only_velocity(kind, x, data, reference, bandwidth):
            nonlocal calls
            _ = kind, data, reference, bandwidth
            calls += 1
            return torch.ones_like(x) if calls == 3 else torch.zeros_like(x)

        try:
            wgf_module.wgf_velocity = arbitrary_only_velocity
            _, stats = fast_wgf_flow_map_loss(
                model,
                z,
                z,
                z,
                dt=0.1,
                path_weight=1.0,
                lambda_boundary=1.0,
            )
        finally:
            wgf_module.wgf_velocity = original_velocity

        self.assertAlmostEqual(float(stats["loss_boundary_origin"]), 0.0, places=6)
        self.assertGreater(float(stats["loss_boundary_arbitrary"]), 0.0)

    def test_fast_boundary_velocity_uses_one_full_batch_condition(self):
        model = ConstantFlowMap([0.1, -0.2])
        z = torch.zeros(4, 2)
        calls = []
        original_velocity = wgf_module.wgf_velocity

        def recording_velocity(kind, x, data, reference, bandwidth):
            _ = kind, data, reference, bandwidth
            calls.append(x.shape[0])
            return torch.ones_like(x)

        try:
            wgf_module.wgf_velocity = recording_velocity
            fast_wgf_flow_map_loss(
                model,
                z,
                z,
                z,
                dt=0.1,
                lambda_boundary=1.0,
                time_groups=2,
            )
        finally:
            wgf_module.wgf_velocity = original_velocity

        self.assertEqual(calls, [2, 2, 4, 4])

    def test_transition_mnist_model_uses_group_normalization(self):
        model = TransitionConditionedMnistConvGenerator(base_channels=8)

        self.assertTrue(
            any(isinstance(layer, torch.nn.GroupNorm) for layer in model.net)
        )
        self.assertFalse(
            any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.net)
        )


if __name__ == "__main__":
    unittest.main()
