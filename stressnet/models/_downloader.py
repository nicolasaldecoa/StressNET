"""Download and cache pre-trained StressNET weights from Google Drive."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import gdown
from platformdirs import user_cache_dir

from ._registry import ModelEntry, get_entry

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(user_cache_dir('stressnet', appauthor=False)) / 'models'


def _cache_dir() -> Path:
    """Return (and create if needed) the local model cache directory."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f'SHA-256 mismatch for {path.name}.\n'
            f'  expected : {expected}\n'
            f'  got      : {actual}\n'
            'The file has been removed. Re-run to download again.'
        )


def _gdrive_url(file_id: str) -> str:
    return f'https://drive.google.com/uc?id={file_id}'


def download_model(name: str, *, force: bool = False) -> Path:
    """Download model weights by *name* and return the local cache path.

    The file is only downloaded when it is absent or *force* is ``True``.
    After downloading, the SHA-256 checksum is verified when one is present
    in the registry entry.

    Parameters
    ----------
    name:
        Model name as registered in ``_registry._REGISTRY``.
    force:
        Re-download even if the file already exists in the cache.

    Returns
    -------
    Path
        Absolute path to the cached weights file.
    """
    entry: ModelEntry = get_entry(name)
    dest = _cache_dir() / entry.filename

    if dest.exists() and not force:
        logger.debug('Model %r already cached at %s', name, dest)
        if entry.sha256:
            _verify(dest, entry.sha256)
        return dest

    logger.info('Downloading model %r from Google Drive…', name)
    url = _gdrive_url(entry.gdrive_id)
    gdown.download(url, str(dest), quiet=False)

    if not dest.exists():
        raise RuntimeError(
            f'Download of model {name!r} failed: file not found at {dest}.'
        )

    if entry.sha256:
        logger.info('Verifying checksum for %r…', name)
        _verify(dest, entry.sha256)

    logger.info('Model %r saved to %s', name, dest)
    return dest


def get_model_path(name: str) -> Path:
    """Return the cached path for *name*, downloading it if necessary."""
    return download_model(name)


def clear_cache(name: str | None = None) -> None:
    """Delete cached weights.

    Parameters
    ----------
    name:
        Model name to remove.  Pass ``None`` to clear the entire cache.
    """
    if name is not None:
        entry = get_entry(name)
        target = _cache_dir() / entry.filename
        if target.exists():
            target.unlink()
            logger.info('Removed cached model %r (%s)', name, target)
    else:
        import shutil
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
        logger.info('Cleared entire StressNET model cache at %s', _CACHE_DIR)
