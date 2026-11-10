"""DDIM eta=0 sampling with exactly NFE calls on the trained DDPM schedule."""
from __future__ import annotations

import torch


def ddim_timesteps(training_timesteps: int, nfe: int) -> list[int]:
    if not 1 <= nfe <= training_timesteps:
        raise ValueError("NFE must be between 1 and the number of training timesteps")
    # Matches the installed lucidrains sampler; T and alpha_bar stay unchanged.
    return list(reversed(torch.linspace(-1, training_timesteps - 1, nfe + 1).int().tolist()))


@torch.inference_mode()
def sample_ddim(diffusion, initial_noise: torch.Tensor, nfe: int) -> torch.Tensor:
    """Return RGB images in [0,1]; caller owns noise and final uint8 quantization."""
    times = ddim_timesteps(diffusion.num_timesteps, nfe)
    image = initial_noise
    for timestep, next_timestep in zip(times[:-1], times[1:]):
        time = torch.full((image.shape[0],), timestep, device=image.device, dtype=torch.long)
        prediction = diffusion.model_predictions(image, time, clip_x_start=True, rederive_pred_noise=True)
        if next_timestep < 0:
            image = prediction.pred_x_start
        else:
            alpha_next = diffusion.alphas_cumprod[next_timestep]
            image = alpha_next.sqrt() * prediction.pred_x_start + (1 - alpha_next).sqrt() * prediction.pred_noise
    return diffusion.unnormalize(image).clamp(0, 1)
