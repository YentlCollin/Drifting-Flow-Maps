"""Secure cache I/O helpers used by the latent experiments."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _sha256_fd(descriptor: int, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _absolute_clean_path(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or any(
        part in {".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"secure path must be absolute and normalized: {path}")
    return candidate


def _open_directory_chain(path: Path) -> int:
    """Open every directory component with ``O_NOFOLLOW`` from filesystem root."""

    candidate = _absolute_clean_path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in candidate.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"secure directory is not a directory: {candidate}")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_file_binding(
    path: Path, parent_descriptor: int, expected: os.stat_result
) -> None:
    """Require both the held parent and the live absolute path to name the held inode."""

    expected_identity = _stable_stat_identity(expected)
    local = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    if _stable_stat_identity(local) != expected_identity or not stat.S_ISREG(
        local.st_mode
    ):
        raise RuntimeError(f"secure file binding changed below held parent: {path}")
    live_parent = _open_directory_chain(path.parent)
    try:
        live = os.stat(path.name, dir_fd=live_parent, follow_symlinks=False)
    finally:
        os.close(live_parent)
    if _stable_stat_identity(live) != expected_identity or not stat.S_ISREG(
        live.st_mode
    ):
        raise RuntimeError(f"secure absolute file binding changed: {path}")


@contextmanager
def secure_regular_fd(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> Iterator[int]:
    """Yield a stable read-only regular-file fd, rejecting every symlink component."""

    candidate = _absolute_clean_path(path)
    if not candidate.name:
        raise ValueError("secure regular-file path has no basename")
    parent_descriptor = _open_directory_chain(candidate.parent)
    descriptor = -1
    try:
        descriptor = os.open(
            candidate.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"secure input is not a regular file: {candidate}")
        if expected_size is not None and (
            type(expected_size) is not int or before.st_size != expected_size
        ):
            raise ValueError(f"secure input byte size differs: {candidate}")
        initial_digest = _sha256_fd(descriptor) if expected_sha256 is not None else None
        if expected_sha256 is not None and initial_digest != expected_sha256:
            raise ValueError(f"secure input SHA-256 differs: {candidate}")
        _assert_file_binding(candidate, parent_descriptor, before)
        try:
            yield descriptor
        finally:
            after = os.fstat(descriptor)
            if _stable_stat_identity(after) != _stable_stat_identity(before):
                raise RuntimeError(
                    f"secure input inode changed while held: {candidate}"
                )
            if initial_digest is not None and _sha256_fd(descriptor) != initial_digest:
                raise RuntimeError(
                    f"secure input bytes changed while held: {candidate}"
                )
            _assert_file_binding(candidate, parent_descriptor, before)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def secure_sha256_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> str:
    with secure_regular_fd(
        path, expected_sha256=expected_sha256, expected_size=expected_size
    ) as descriptor:
        return _sha256_fd(descriptor)


def secure_directory_metadata(path: Path) -> dict[str, int]:
    """Return directory metadata while holding a no-follow descriptor chain."""

    target = require_real_directory(path)
    descriptor = _open_directory_chain(target)
    try:
        before = os.fstat(descriptor)
        live = _open_directory_chain(target)
        try:
            current = os.fstat(live)
        finally:
            os.close(live)
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"secure directory binding changed: {target}")
        return {
            "device": before.st_dev,
            "inode": before.st_ino,
            "mode": stat.S_IMODE(before.st_mode),
        }
    finally:
        os.close(descriptor)


@contextmanager
def secure_directory_fd(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> Iterator[int]:
    """Hold one no-follow directory fd and continuously bind it to ``path``.

    Directory mtime/ctime necessarily change when children are staged, so the
    durable identity used by callers is the exact ``(st_dev, st_ino)`` pair.
    """

    target = _absolute_clean_path(path)
    descriptor = _open_directory_chain(target)
    try:
        before = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise RuntimeError(f"secure directory identity differs: {target}")
        try:
            yield descriptor
        finally:
            after = os.fstat(descriptor)
            if (after.st_dev, after.st_ino) != identity:
                raise RuntimeError(f"held directory identity changed: {target}")
            live = _open_directory_chain(target)
            try:
                live_metadata = os.fstat(live)
            finally:
                os.close(live)
            if (live_metadata.st_dev, live_metadata.st_ino) != identity:
                raise RuntimeError(
                    f"secure directory binding changed while held: {target}"
                )
    finally:
        os.close(descriptor)


def _validate_child_name(name: str) -> bytes:
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError(f"unsafe directory child name: {name!r}")
    return os.fsencode(name)


def secure_publish_regular_noreplace_at(
    directory_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    expected_source_identity: tuple[int, int],
) -> None:
    """Atomically publish one regular leaf without replacing a destination.

    Linux ``RENAME_NOREPLACE`` is preferred.  Some production NFS servers
    reject rename flags with ``EINVAL``; on those filesystems a same-directory
    hard-link create followed by unlink of the staging name has the same
    create-new property for regular files.  The transient second link is
    inode-verifiable and recoverable after SIGKILL.
    """

    source_encoded = _validate_child_name(source_name)
    destination_encoded = _validate_child_name(destination_name)
    parent_metadata = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("publication dirfd is not a directory")
    source = os.stat(source_name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(source.st_mode)
        or (source.st_dev, source.st_ino) != expected_source_identity
    ):
        raise RuntimeError("regular publication source identity differs")
    try:
        os.stat(destination_name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            f"publication destination already exists: {destination_name}"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    result = -1
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            directory_descriptor,
            source_encoded,
            directory_descriptor,
            destination_encoded,
            1,
        )
    if result != 0:
        error_number = errno.ENOSYS if renameat2 is None else ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                f"publication destination raced into existence: {destination_name}"
            )
        unsupported = {
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            getattr(errno, "ENOTSUP", errno.EINVAL),
        }
        if error_number not in unsupported:
            raise OSError(error_number, os.strerror(error_number), destination_name)
        try:
            os.link(
                source_name,
                destination_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"publication destination raced into existence: {destination_name}"
            ) from error
        linked = os.stat(
            destination_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(linked.st_mode)
            or (linked.st_dev, linked.st_ino) != expected_source_identity
        ):
            raise RuntimeError("hard-link publication identity differs")
        os.fsync(directory_descriptor)
        source_after_link = os.stat(
            source_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            source_after_link.st_dev,
            source_after_link.st_ino,
        ) != expected_source_identity:
            raise RuntimeError("hard-link publication source changed before unlink")
        os.unlink(source_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    published = os.stat(
        destination_name, dir_fd=directory_descriptor, follow_symlinks=False
    )
    if (
        not stat.S_ISREG(published.st_mode)
        or (published.st_dev, published.st_ino) != expected_source_identity
    ):
        raise RuntimeError("published regular-file identity differs")
    try:
        os.stat(source_name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("regular publication source still exists")
    os.fsync(directory_descriptor)


def secure_unlink_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Unlink exactly one regular child from a held directory descriptor."""

    _validate_child_name(name)
    metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise RuntimeError("refusing unlink of a replaced/non-regular child")
    os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


def require_real_directory(path: Path) -> Path:
    candidate = _absolute_clean_path(path)
    descriptor = _open_directory_chain(candidate)
    try:
        held = os.fstat(descriptor)
        live_descriptor = _open_directory_chain(candidate)
        try:
            live = os.fstat(live_descriptor)
        finally:
            os.close(live_descriptor)
        if (held.st_dev, held.st_ino) != (live.st_dev, live.st_ino):
            raise RuntimeError(f"secure directory binding changed: {candidate}")
    finally:
        os.close(descriptor)
    return candidate
