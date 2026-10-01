"""Sphinx configuration for the TSGap documentation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "TSGap"
author = "Feruz Oripov"
copyright = "2026, Feruz Oripov"
release = "0.7.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "TSGap documentation"

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
myst_heading_anchors = 3
