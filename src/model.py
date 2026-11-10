import math

import torch
import torch.nn as nn

try:
    from denoising_diffusion_pytorch import Unet as LucidrainsUNet
except ModuleNotFoundError as exc:
    if exc.name != "denoising_diffusion_pytorch":
        raise
    LucidrainsUNet = None

MNIST_DIM = 28 * 28
CELEBA_CHANNELS = 3
CELEBA_IMAGE_SIZE = 64
CELEBA_DIM = CELEBA_CHANNELS * CELEBA_IMAGE_SIZE * CELEBA_IMAGE_SIZE


def prepare_time_features(
    t,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a `[batch_size, 1]` time column compatible with latent batches."""
    t = torch.as_tensor(t, device=device, dtype=dtype)
    if t.ndim == 0:
        return t.expand(batch_size, 1)
    return t.reshape(batch_size, 1)


def prepare_transition_features(
    t,
    s,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t_features = prepare_time_features(
        t,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    s_features = prepare_time_features(
        s,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    return torch.cat([t_features, s_features], dim=1)


class MLP(nn.Module):
    """MLP: noise -> output. 3 hidden layers with SiLU."""

    def __init__(self, in_dim=2, hidden=256, out_dim=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class TimeConditionedMLP(nn.Module):
    """Residual flow map `T(z, t) = z + t f(z, t)`."""

    def __init__(self, in_dim=2, hidden=256, out_dim=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, t):
        t_features = prepare_time_features(
            t,
            batch_size=z.shape[0],
            device=z.device,
            dtype=z.dtype,
        )
        return (1 - t_features) * z + t_features * self.net(torch.cat([z, t_features], dim=1))


class TransitionConditionedMLP(nn.Module):
    """Two-time flow map `F(x, t, s) = x + (s - t) v(x, t, s)`."""

    def __init__(self, in_dim=2, hidden=256, out_dim=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, t, s):
        transition_features = prepare_transition_features(
            t,
            s,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        delta_t = transition_features[:, 1:] - transition_features[:, :1]
        return x + delta_t * self.net(torch.cat([x, transition_features], dim=1))


def _mnist_decoder(
    base_channels: int,
    *,
    group_norm: bool = False,
) -> nn.Sequential:
    c1 = base_channels * 4
    c2 = base_channels * 2
    c3 = base_channels
    normalization = (
        lambda channels: nn.GroupNorm(math.gcd(8, channels), channels)
        if group_norm
        else nn.BatchNorm2d(channels)
    )
    return nn.Sequential(
        nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),  # 7 -> 14
        normalization(c2),
        nn.SiLU(),
        nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),  # 14 -> 28
        normalization(c3),
        nn.SiLU(),
        nn.Conv2d(c3, 1, kernel_size=3, stride=1, padding=1),
    )


class MnistConvGenerator(nn.Module):
    """Convolutional generator: z -> 28x28 MNIST image (flattened output)."""

    def __init__(self, latent_dim=128, base_channels=64):
        super().__init__()
        c1 = base_channels * 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, c1 * 7 * 7),
            nn.LayerNorm(c1 * 7 * 7),
            nn.SiLU(),
        )
        self.net = _mnist_decoder(base_channels)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.shape[0], -1, 7, 7)
        x = self.net(x)
        x = torch.sigmoid(x)  # match MNIST range [0,1]
        return x.view(z.shape[0], -1)


class TimeConditionedMnistConvGenerator(nn.Module):
    """Time-conditioned convolutional generator for MNIST."""

    def __init__(self, latent_dim=MNIST_DIM, base_channels=64, time_embed_dim=128):
        super().__init__()
        if latent_dim != MNIST_DIM:
            raise ValueError(f"latent_dim must be {MNIST_DIM} for TimeConditionedMnistConvGenerator")
        c1 = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, c1 * 7 * 7),
            nn.LayerNorm(c1 * 7 * 7),
            nn.SiLU(),
        )
        self.net = _mnist_decoder(base_channels)

    def forward(self, z, t):
        t_features = prepare_time_features(
            t,
            batch_size=z.shape[0],
            device=z.device,
            dtype=z.dtype,
        )
        t_embedding = self.time_embed(t_features)
        x = self.fc(torch.cat([z, t_embedding], dim=1))
        x = x.view(z.shape[0], -1, 7, 7)
        x = self.net(x)
        x = torch.sigmoid(x).view(z.shape[0], -1)
        return (1 - t_features) * z + t_features * x


class TimeConditionedMnistDenoiser(TimeConditionedMnistConvGenerator):
    """Time-conditioned MNIST x0 denoiser."""

    def forward(self, x, t):
        t_features = prepare_time_features(
            t,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        t_embedding = self.time_embed(t_features)
        out = self.fc(torch.cat([x, t_embedding], dim=1))
        out = out.view(x.shape[0], -1, 7, 7)
        return torch.sigmoid(self.net(out)).view(x.shape[0], -1)


class TransitionConditionedMnistConvGenerator(nn.Module):
    """Two-time bounded endpoint map for MNIST states."""

    def __init__(
        self,
        latent_dim=MNIST_DIM,
        base_channels=64,
        time_embed_dim=128,
        group_norm=True,
    ):
        super().__init__()
        if latent_dim != MNIST_DIM:
            raise ValueError(f"latent_dim must be {MNIST_DIM} for TransitionConditionedMnistConvGenerator")
        c1 = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, c1 * 7 * 7),
            nn.LayerNorm(c1 * 7 * 7),
            nn.SiLU(),
        )
        self.net = _mnist_decoder(base_channels, group_norm=group_norm)

    def forward(self, x, t, s):
        transition_features = prepare_transition_features(
            t,
            s,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        t_embedding = self.time_embed(transition_features)
        out = self.fc(torch.cat([x, t_embedding], dim=1))
        out = out.view(x.shape[0], -1, 7, 7)
        out = self.net(out)
        endpoint = torch.sigmoid(out).view(x.shape[0], -1)
        start_t = transition_features[:, :1]
        delta_t = (transition_features[:, 1:] - start_t).clamp_min(0.0)
        remaining_t = (1.0 - start_t).clamp_min(1e-3)
        weight = (delta_t / remaining_t).clamp(max=1.0)
        return (1 - weight) * x + weight * endpoint


class _TransitionTimeEmbedding(nn.Module):
    def __init__(self, sinusoidal_embedding: nn.Module, dim: int):
        super().__init__()
        time_dim = dim * 4
        self.sinusoidal_embedding = sinusoidal_embedding
        self.projection = nn.Sequential(
            nn.Linear(dim * 2, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, transition_features: torch.Tensor) -> torch.Tensor:
        start_t, end_t = transition_features.unbind(dim=1)
        return self.projection(
            torch.cat(
                [
                    self.sinusoidal_embedding(start_t),
                    self.sinusoidal_embedding(end_t),
                ],
                dim=1,
            )
        )


class TimeConditionedCelebaUNet(nn.Module):
    """Time-conditioned CelebA denoiser."""

    def __init__(self, latent_dim=CELEBA_DIM, base_channels=64):
        super().__init__()
        if LucidrainsUNet is None:
            raise ImportError("denoising-diffusion-pytorch is required for CelebA models")
        if latent_dim != CELEBA_DIM:
            raise ValueError(f"latent_dim must be {CELEBA_DIM} for TimeConditionedCelebaUNet")
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for the lucidrains U-Net")
        self.net = LucidrainsUNet(
            dim=base_channels,
            dim_mults=(1, 2, 4),
            channels=CELEBA_CHANNELS,
            out_dim=CELEBA_CHANNELS,
        )

    def forward(self, x, t):
        t_features = prepare_time_features(
            t,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        image = x.view(x.shape[0], CELEBA_CHANNELS, CELEBA_IMAGE_SIZE, CELEBA_IMAGE_SIZE)
        out = torch.sigmoid(self.net(image, t_features[:, 0])).view(x.shape[0], -1)
        return out


class TransitionConditionedCelebaUNet(nn.Module):
    """Lucidrains U-Net adapted as a two-time endpoint map for CelebA."""

    def __init__(
        self,
        latent_dim=CELEBA_DIM,
        base_channels=64,
    ):
        super().__init__()
        if LucidrainsUNet is None:
            raise ImportError("denoising-diffusion-pytorch is required for CelebA models")
        if latent_dim != CELEBA_DIM:
            raise ValueError(f"latent_dim must be {CELEBA_DIM} for TransitionConditionedCelebaUNet")
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for the lucidrains U-Net")
        self.net = LucidrainsUNet(
            dim=base_channels,
            dim_mults=(1, 2, 4),
            channels=CELEBA_CHANNELS,
            out_dim=CELEBA_CHANNELS,
        )
        self.net.time_mlp = _TransitionTimeEmbedding(
            self.net.time_mlp[0],
            base_channels,
        )

    def forward(self, x, t, s):
        transition_features = prepare_transition_features(
            t,
            s,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        image = x.view(x.shape[0], CELEBA_CHANNELS, CELEBA_IMAGE_SIZE, CELEBA_IMAGE_SIZE)
        out = torch.sigmoid(self.net(image, transition_features)).view(x.shape[0], -1)
        start_t = transition_features[:, :1]
        delta_t = (transition_features[:, 1:] - start_t).clamp_min(0.0)
        remaining_t = (1.0 - start_t).clamp_min(1e-3)
        weight = (delta_t / remaining_t).clamp(max=1.0)
        return (1 - weight) * x + weight * out


class CelebaUNetGenerator(TransitionConditionedCelebaUNet):
    """Endpoint-only CelebA generator for the optimizer-time drift baseline."""

    def forward(self, x):
        t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return super().forward(x, t, torch.ones_like(t))
