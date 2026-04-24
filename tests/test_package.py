"""Smoke tests for the public package surface."""

import stressnet


def test_import() -> None:
    assert stressnet is not None


def test_version() -> None:
    version_parts = stressnet.__version__.split(".")
    assert len(version_parts) == 3
    assert all(part.isdigit() for part in version_parts)


def test_public_exports_are_available() -> None:
    missing = [name for name in stressnet.__all__ if not hasattr(stressnet, name)]
    assert missing == []
