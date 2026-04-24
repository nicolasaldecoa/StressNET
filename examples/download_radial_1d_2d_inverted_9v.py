from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

DATASET_URL = 'https://drive.google.com/file/d/18njjkc0Ea1ITzqxb33vgKBC5cJvrLLR0/view?usp=sharing'
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / 'example_data'
    / 'finetuning'
    / 'radial_1d_2d_inverted_9v'
)


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            member_path = destination / member
            resolved_member = member_path.resolve()
            if not str(resolved_member).startswith(str(destination) + os.sep):
                raise RuntimeError(f'Unsafe zip member path detected: {member}')
        zf.extractall(destination)


def _gdrive_direct_url_from_share_url(share_url: str) -> str:
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', share_url)
    if not match:
        raise ValueError(f'Could not parse Google Drive file id from URL: {share_url}')
    file_id = match.group(1)
    return f'https://drive.google.com/uc?id={file_id}'


def _dataset_layout_complete(output: Path) -> bool:
    """True when ``output/train`` and ``output/test`` are both present (expected layout)."""
    return (output / 'train').is_dir() and (output / 'test').is_dir()


def ensure_dataset_downloaded(output: Path = DEFAULT_OUTPUT, force: bool = False) -> None:
    """Download and extract the radial_1d_2d_inverted_9v dataset.

    Skips download when ``output/train`` and ``output/test`` already exist unless
    ``force`` is True. After a successful extract, validates that layout.

    This function is notebook-friendly and can be imported/called directly.
    """
    output = Path(output)
    if output.exists():
        if not force and _dataset_layout_complete(output):
            print('Dataset already available. Use force=True (or --force flag) to delete and download again.')
            return
        shutil.rmtree(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    import gdown

    with tempfile.TemporaryDirectory(prefix='stressnet_dataset_') as tmp_dir:
        tmp_zip_path = Path(tmp_dir) / 'radial_1d_2d_inverted_9v.zip'
        direct_url = _gdrive_direct_url_from_share_url(DATASET_URL)
        downloaded_zip = gdown.download(
            direct_url,
            str(tmp_zip_path),
            quiet=False,
        )

        if not downloaded_zip:
            raise RuntimeError('Download failed: zip file was not downloaded.')

        extract_root = Path(tmp_dir) / 'extract'
        extract_root.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(zip_path=tmp_zip_path, destination=extract_root)

        # Support either:
        # 1) zip contains radial_1d_2d_inverted_9v/...
        # 2) zip contains train/, test/, ... directly
        nested_root = extract_root / output.name
        source_dir = nested_root if nested_root.exists() else extract_root
        output.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            shutil.move(str(item), str(output / item.name))

    if not output.exists():
        raise RuntimeError(f'Download finished but expected folder was not created: {output}')
    if not _dataset_layout_complete(output):
        raise RuntimeError(
            f'After extract, expected {output / "train"} and {output / "test"} to exist. '
            'Check the zip layout or use --force to retry.'
        )
    print(f'Dataset downloaded to: {output}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download radial_1d_2d_inverted_9v example dataset from Google Drive.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Target dataset directory (default: examples/example_data/finetuning/radial_1d_2d_inverted_9v).',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete target folder first and re-download.',
    )
    args = parser.parse_args()
    ensure_dataset_downloaded(output=args.output, force=args.force)


if __name__ == '__main__':
    main()
