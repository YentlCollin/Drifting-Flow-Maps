from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)
