import hashlib

import pytest

import stressnet.models._downloader as downloader


def test_gdrive_url_builds_direct_download_url() -> None:
    assert downloader._gdrive_url('abc123') == 'https://drive.google.com/uc?id=abc123'


def test_sha256_hashes_file_contents(tmp_path) -> None:
    path = tmp_path / 'payload.bin'
    payload = b'stressnet'
    path.write_bytes(payload)

    assert downloader._sha256(path) == hashlib.sha256(payload).hexdigest()


def test_verify_removes_file_on_checksum_mismatch(tmp_path) -> None:
    path = tmp_path / 'payload.bin'
    path.write_bytes(b'bad-data')

    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        downloader._verify(path, expected='not-the-real-digest')

    assert not path.exists()


def test_download_model_uses_cache_without_gdown(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(downloader, '_cache_dir', lambda: tmp_path)
    monkeypatch.setattr(downloader, '_verify', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        downloader.gdown,
        'download',
        lambda *_args, **_kwargs: pytest.fail('download should not be called for cache hit'),
    )
    cached = tmp_path / 'StressNET_weights_pretrained_A.h5'
    cached.write_bytes(b'cached')

    path = downloader.download_model('STRESSNET-PRETRAINED-A')

    assert path == cached


def test_download_model_download_failure_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(downloader, '_cache_dir', lambda: tmp_path)
    monkeypatch.setattr(downloader.gdown, 'download', lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match='file not found'):
        downloader.download_model('STRESSNET-PRETRAINED-A', force=True)


def test_download_model_downloads_and_verifies(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(downloader, '_cache_dir', lambda: tmp_path)
    calls = {}

    def fake_download(url: str, dest: str, quiet: bool = False) -> None:
        calls['download'] = {'url': url, 'dest': dest, 'quiet': quiet}
        tmp_path.joinpath('StressNET_weights_pretrained_A.h5').write_bytes(b'downloaded')

    def fake_verify(path, expected):
        calls['verify'] = {'path': path, 'expected': expected}

    monkeypatch.setattr(downloader.gdown, 'download', fake_download)
    monkeypatch.setattr(downloader, '_verify', fake_verify)

    path = downloader.download_model('STRESSNET-PRETRAINED-A', force=True)

    assert path == tmp_path / 'StressNET_weights_pretrained_A.h5'
    assert calls['download']['url'] == 'https://drive.google.com/uc?id=1Uk5FGMctLNePXhHX_dfWZq-jOUW5OMIJ'
    assert calls['download']['dest'] == str(path)
    assert calls['download']['quiet'] is False
    assert calls['verify']['path'] == path
