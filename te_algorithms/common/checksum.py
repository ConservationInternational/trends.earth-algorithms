"""Checksums for local files."""

import hashlib


def file_checksum(path):
    """Return the SHA-256 hexadecimal digest of a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
