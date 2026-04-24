#!/usr/bin/env python3
"""Build HTML documentation locally (cross-platform; no Makefile required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    docs_dir = Path(__file__).resolve().parent
    out_dir = docs_dir / '_build' / 'html'
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        '-m',
        'sphinx.cmd.build',
        '-b',
        'html',
        str(docs_dir),
        str(out_dir),
    ]
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
