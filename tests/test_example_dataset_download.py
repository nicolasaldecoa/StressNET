import sys
import types
import zipfile
from pathlib import Path

import pytest

from examples import download_radial_1d_2d_inverted_9v as downloader


def _make_dataset_zip(path: Path, *, nested: bool = False) -> None:
    prefix = 'radial_1d_2d_inverted_9v/' if nested else ''
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr(f'{prefix}train/example_train.npz', b'train')
        zf.writestr(f'{prefix}test/example_test.npz', b'test')


def test_gdrive_direct_url_from_share_url() -> None:
    share_url = 'https://drive.google.com/file/d/18njjkc0Ea1ITzqxb33vgKBC5cJvrLLR0/view?usp=sharing'

    direct_url = downloader._gdrive_direct_url_from_share_url(share_url)

    assert direct_url == 'https://drive.google.com/uc?id=18njjkc0Ea1ITzqxb33vgKBC5cJvrLLR0'


def test_gdrive_direct_url_rejects_malformed_url() -> None:
    with pytest.raises(ValueError, match='Could not parse Google Drive file id'):
        downloader._gdrive_direct_url_from_share_url('https://drive.google.com/drive/folders/folder-id')


def test_safe_extract_zip_rejects_path_traversal(tmp_path) -> None:
    zip_path = tmp_path / 'unsafe.zip'
    destination = tmp_path / 'out'
    destination.mkdir()
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('../evil.txt', 'nope')

    with pytest.raises(RuntimeError, match='Unsafe zip member path'):
        downloader._safe_extract_zip(zip_path, destination)


def test_dataset_layout_complete_requires_train_and_test(tmp_path) -> None:
    assert not downloader._dataset_layout_complete(tmp_path)
    (tmp_path / 'train').mkdir()
    assert not downloader._dataset_layout_complete(tmp_path)
    (tmp_path / 'test').mkdir()
    assert downloader._dataset_layout_complete(tmp_path)


def test_ensure_dataset_downloaded_short_circuits_when_layout_exists(monkeypatch, tmp_path) -> None:
    (tmp_path / 'train').mkdir()
    (tmp_path / 'test').mkdir()

    fake_gdown = types.SimpleNamespace(download=lambda *_args, **_kwargs: pytest.fail('unexpected download'))
    monkeypatch.setitem(sys.modules, 'gdown', fake_gdown)

    assert downloader.ensure_dataset_downloaded(output=tmp_path) is None


def test_ensure_dataset_downloaded_extracts_mocked_zip(monkeypatch, tmp_path) -> None:
    fixture_zip = tmp_path / 'fixture.zip'
    output = tmp_path / 'radial_1d_2d_inverted_9v'
    _make_dataset_zip(fixture_zip, nested=True)

    def fake_download(_url: str, output_path: str, quiet: bool = False) -> str:
        Path(output_path).write_bytes(fixture_zip.read_bytes())
        return output_path

    monkeypatch.setitem(sys.modules, 'gdown', types.SimpleNamespace(download=fake_download))

    assert downloader.ensure_dataset_downloaded(output=output) is None
    assert (output / 'train' / 'example_train.npz').is_file()
    assert (output / 'test' / 'example_test.npz').is_file()
