"""Use canonical temporary paths for secure-I/O and multiprocessing tests."""

import os
from pathlib import Path
import sys
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def canonical_temporary_directory():
    # macOS /var and /tmp are symlinks; PyTorch also needs short socket paths.
    parent = Path("/private/tmp") if sys.platform == "darwin" else Path(tempfile.gettempdir()).resolve()
    previous_cache = tempfile.tempdir
    previous_env = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(prefix="flowmap-", dir=parent) as directory:
        directory = str(Path(directory).resolve())
        tempfile.tempdir = directory
        os.environ["TMPDIR"] = directory
        try:
            yield
        finally:
            tempfile.tempdir = previous_cache
            if previous_env is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_env
