"""Registry of pre-trained StressNET model weights hosted on Google Drive.

Each entry maps a model name to its download metadata.  To register a new
model, add an entry to ``_REGISTRY`` following the same structure.

Fields
------
gdrive_id : str
    The Google Drive file ID extracted from the sharing URL
    ``https://drive.google.com/file/d/<gdrive_id>/view``.
filename : str
    Local filename used when saving to the cache directory.
sha256 : str | None
    Hex-encoded SHA-256 digest of the file.  Set to ``None`` to skip
    integrity checking (not recommended for published weights).
description : str
    Short human-readable description shown by ``list_models()``.
points_per_edge : int
    Value of ``points_per_edge`` expected by ``build_model`` for this
    checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    gdrive_id: str
    filename: str
    description: str
    points_per_edge: int
    sha256: str


_REGISTRY: dict[str, ModelEntry] = {
    'STRESSNET-PRETRAINED-A': ModelEntry(
        gdrive_id='1Uk5FGMctLNePXhHX_dfWZq-jOUW5OMIJ',
        filename='StressNET_weights_pretrained_A.h5',
        sha256='fd133b6e7fa7569db0ac9e79fc7d10ff2a91f7a915d694499cb4e686460a2502',
        description='StressNET pretrained model (replica A)',
        points_per_edge=9,
    ),
    'STRESSNET-PRETRAINED-B': ModelEntry(
        gdrive_id='1n_ZMn4QFInO6DS5MEeuOEo4dk9pyU2iI',
        filename='StressNET_weights_pretrained_B.h5',
        sha256='abd78a6fc9401d057fd8319a4c0bd57775cf7e401f89706d4ca993cae379cc2c',
        description='StressNET pretrained model (replica B)',
        points_per_edge=9,
    ),
    'STRESSNET-PRETRAINED-C': ModelEntry(
        gdrive_id='1SvNh4FJGBDyOIA1JYrFXzHfgVDFRqmv-',
        filename='StressNET_weights_pretrained_C.h5',
        sha256='20a8202b4722c2eca193374517afa873a06ab2658897a72e5e71c79f39ffed47',
        description='StressNET pretrained model (replica C)',
        points_per_edge=9,
    ),
    'STRESSNET-PRETRAINED-D': ModelEntry(
        gdrive_id='16Fo4uctlJhQoh12fapp5hifgvcx9cJEK',
        filename='StressNET_weights_pretrained_D.h5',
        sha256='399c6abad1c25ca55dea51397e99de3bc95a1624fce831f43f74ae973b19aca3',
        description='StressNET pretrained model (replica D)',
        points_per_edge=9,
    ),
    'STRESSNET-PRETRAINED-E': ModelEntry(
        gdrive_id='1JDzz4K7A7TqwZlsLObEMcvsWkTUfMpn5',
        filename='StressNET_weights_pretrained_E.h5',
        sha256='0018a81a55e88a47c0537526543d54526ea6bff7734b64237329e571e7641d3a',
        description='StressNET pretrained model (replica E)',
        points_per_edge=9,
    ),
    'STRESSNET-FINETUNED-MYOSIN-88-A': ModelEntry(
        gdrive_id='1PEiscpCaeLdVjR6qE2IsVVEzFFUf0PkI',
        filename='StressNET_weights_finetuned_myosin_88_A.h5',
        sha256='b44255964a80d9e211e7703dc8ff4a53f99ec084ddb90df0cb97cb81a8779bd8',
        description='StressNET model (replica A) finetuned on 88 myosin samples',
        points_per_edge=9,
    ),
}


def get_entry(name: str
              ) -> ModelEntry:
    """Return the registry entry for *name*, raising ``KeyError`` if absent."""
    name = name.upper()
    if name not in _REGISTRY:
        available = ', '.join(_REGISTRY) or '(none registered yet)'
        raise KeyError(f'Unknown model {name!r}. Available models: {available}')
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return list(_REGISTRY.keys())
