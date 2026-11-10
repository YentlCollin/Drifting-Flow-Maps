from functools import partial

import torch


def q_sample(x_0, t, alphas, sigmas):
    shape = (x_0.shape[0],) + (1,) * (x_0.ndim - 1)
    return (
        alphas[t].view(shape) * x_0
        + sigmas[t].view(shape) * torch.randn_like(x_0)
    )


def model_denoiser(model, x_t, t, alphas, sigmas):
    t = torch.as_tensor(t, device=x_t.device)
    if t.ndim == 0:
        t = t.expand(x_t.shape[0])
    return model(x_t, t.to(x_t.dtype) / (len(alphas) - 1))


def bridge_statistics(x_t, x_0_hat, s, t, alphas, sigmas, eta_fn):
    """
    s < t
    """
    ddim_std = eta_fn(alphas=alphas, sigmas=sigmas, s=s, t=t)
    x1_scale = (sigmas[s] ** 2 - ddim_std**2).sqrt()
    mean = alphas[s] * x_0_hat + x1_scale * (x_t - alphas[t] * x_0_hat) / sigmas[t]
    return mean, ddim_std


def ddim_step(x_t, x_0_hat, s, t, alphas, sigmas, eta_fn):
    mean, std = bridge_statistics(
        x_t=x_t, x_0_hat=x_0_hat, s=s, t=t, alphas=alphas, sigmas=sigmas, eta_fn=eta_fn
    )
    return mean + std * torch.randn_like(x_t)


def ddim(initial_noise, denoiser_fn, n_steps, eta, **schedule_kwargs):
    alphas, sigmas = mk_schedule(n_steps=n_steps, **schedule_kwargs)
    alphas = alphas.to(initial_noise.device)
    sigmas = sigmas.to(initial_noise.device)
    eta_fn = partial(mk_eta_fn, eta=eta)
    schedule_type = schedule_kwargs["schedule_type"]

    x_t = initial_noise
    if schedule_type == "ve":
        x_t *= sigmas[-1]

    for t in range(n_steps - 1, 0, -1):
        x0_hat = denoiser_fn(x_t, t, alphas, sigmas)
        if t - 1 == 0:
            return x0_hat
        else:
            x_t = ddim_step(x_t, x0_hat, t - 1, t, alphas, sigmas, eta_fn=eta_fn)


def mk_schedule(schedule_type, **kwargs):
    """
    outputs any type of schedule
    """
    if kwargs["n_steps"] < 2:
        raise ValueError("n_steps must be at least 2")
    if schedule_type == "vp":
        return _mk_vp_schedule(**kwargs)
    elif schedule_type == "fm":
        return _mk_fm_schedule(**kwargs)
    elif schedule_type == "ve":
        return _mk_ve_schedule(**kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def mk_eta_fn(alphas, sigmas, eta, s, t):
    var = (
        sigmas[s].square()
        * (sigmas[t].square() - (alphas[t] * sigmas[s] / alphas[s]).square())
        / sigmas[t].square()
    )
    eta_t = eta * torch.abs(var).sqrt()
    return eta_t


def _mk_vp_schedule(beta_min, beta_max, n_steps, **kwargs):
    alphas = 1 - torch.linspace(beta_min, beta_max, n_steps)
    alpha_bars = torch.cumprod(alphas, dim=0).clamp(min=1e-5, max=1.0 - 1e-5)
    alpha_new = torch.sqrt(alpha_bars)
    sigmas = torch.sqrt(1 - alpha_bars)
    return alpha_new, sigmas


def _mk_fm_schedule(n_steps, **kwargs):
    alphas = torch.linspace(1.0, 0, n_steps).clamp(min=1e-5, max=1.0 - 1e-5)
    sigmas = 1 - alphas
    return alphas, sigmas


def _mk_ve_schedule(sigma_min, sigma_max, rho, n_steps, **kwargs):
    sigmas = (
        sigma_max ** (1 / rho)
        + torch.linspace(1, 0, n_steps)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    alphas = torch.ones_like(sigmas)
    return alphas, sigmas


def diffusion_loss(x_0, x_t, t, alphas, sigmas, denoiser_fn):
    x_0_hat = denoiser_fn(x_t, t, alphas, sigmas)
    return torch.mean((x_0 - x_0_hat) ** 2)
