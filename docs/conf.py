# Sphinx configuration for StressNET. https://www.sphinx-doc.org/

from __future__ import annotations

import os
import sys

# Repo root (parent of docs/)
sys.path.insert(0, os.path.abspath('..'))

# --- Project metadata -----------------------------------------------------
project = 'StressNET'
copyright = '2026, Nicolás Aldecoa Rodrigo and contributors'  # noqa: A001
author = 'Nicolás Aldecoa Rodrigo, ...'

release = '0.1.0'
version = release

# --- Extensions -----------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# NumPy-style docstrings (matches existing module docs in stressnet.models).
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc: avoid importing heavy optional deps on RTD / docs-only CI.
autodoc_mock_imports = [
    'tensorflow',
    'tensorflow.keras',
    'keras',
    'spektral',
    'spektral.layers',
    'spektral.utils',
    'forsys',
    'forsys.frames',
    'matplotlib',
    'matplotlib.pyplot',
    'cmocean',
]

autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
