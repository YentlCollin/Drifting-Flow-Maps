"""Reference asset identities used by the reported RGB continuation."""

from __future__ import annotations

from pathlib import Path

from flowmaps.paths import REPO_ROOT

TRAIN_DIR = REPO_ROOT / "data/celeba64_official_bilinear/training"


TRAIN_FILES_SHA256 = "79b3854e02d591e25eede33f38690d16a1faa6aac57ca77c66844156975c2205"


MAE_PATH = (
    REPO_ROOT
    / "runs/celeba_pixel_gmd/pixel_mae_official_train_base64_seed42_192ep_v1/feature_encoder_ema_final.pt"
)


MAE_SHA256 = "9d088740c9cf725809a3d80b124591dc2f229d3c7772b177257320e84bb7d985"


MAE_SIGNATURE_SHA256 = (
    "fff1d679590be76f3b2089bda88d32c496f591dd27e41ae66451c7ebc230f621"
)
