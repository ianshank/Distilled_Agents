"""Safe archive extraction (blocks path traversal)."""

from __future__ import annotations

import tarfile
from pathlib import Path


def is_safe_tar_member(dest: Path, member_name: str) -> bool:
    """Return True when member_name resolves inside dest."""
    destination = dest.resolve()
    member_path = (destination / member_name).resolve()
    try:
        member_path.relative_to(destination)
    except ValueError:
        return False
    return True


def safe_extract_tar(archive: Path, dest: Path) -> None:
    """Extract a tar archive, rejecting members that escape dest."""
    destination = dest.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if not is_safe_tar_member(destination, member.name):
                raise ValueError(f"Blocked path traversal in archive member: {member.name}")
            tar.extract(member, destination)
