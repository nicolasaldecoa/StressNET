"""Pre-trained model registry and downloader for StressNET.

Quick start
-----------
>>> import stressnet
>>> stressnet.list_models()          # show available model names
>>> model = stressnet.load_model()   # download + build

Lower-level access
------------------
>>> from stressnet.models import get_model_path, clear_cache, list_models
>>> path = get_model_path(list_models()[0])  # returns local Path, downloads if needed
>>> clear_cache(list_models()[0])            # remove cached weights for that name
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._downloader import clear_cache, download_model, get_model_path
from ._registry import list_models

if TYPE_CHECKING:
    import tensorflow as tf


DEFAULT_MODEL = 'STRESSNET-PRETRAINED-A'


def load_model(
    name: str | None = None,
    *,
    force_download: bool = False,
    device: str | None = None,
    **net_kwargs: Any,
) -> tf.keras.Model:
    """Download (if needed) and return a ready-to-use StressNET model.

    Parameters
    ----------
    name:
        Model name as listed by :func:`list_models`.
    force_download:
        Re-download the weights even if they are already cached.
    device:
        TensorFlow device string (e.g. ``'/GPU:0'``).  Defaults to GPU when
        available, otherwise CPU.
    **net_kwargs:
        Extra keyword arguments forwarded to :func:`stressnet.build_stressnet`.

    Returns
    -------
    tf.keras.Model
        Compiled model with weights loaded, ready for inference.
    """
    from ..net.stressnet import get_model
    from ._registry import get_entry

    if name is None:
        name = DEFAULT_MODEL

    entry = get_entry(name)
    weights_path: Path = download_model(name, force=force_download)

    return get_model(
        weights_path=str(weights_path),
        points_per_edge=entry.points_per_edge,
        device=device,
        **net_kwargs,
    )


__all__: list[str] = [
    'list_models',
    'get_model_path',
    'download_model',
    'clear_cache',
    'load_model',
]
