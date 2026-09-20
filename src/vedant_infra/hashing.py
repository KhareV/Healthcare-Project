"""Deterministic SHA-256 helpers for exact artifact bytes."""

from hashlib import sha256
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of *content*."""

    return sha256(content).hexdigest()


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Hash exact file bytes without content normalization."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: str) -> bool:
    """Return whether *value* is a lowercase 64-character SHA-256 digest."""

    if len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)

