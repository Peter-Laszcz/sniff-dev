"""Sphinx configuration for the SNIFF documentation."""

import sys
from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

# The docs build normally runs against installed (editable) packages, but keep
# the source trees importable so `sphinx-build docs docs/_build/html` also works
# in a checkout where nothing has been pip-installed.
_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("sniff-core", "sniff-gui"):
    sys.path.insert(0, str(_ROOT / "packages" / _pkg / "src"))

project = "SNIFF"
author = "Peter Laszcz, Scott Young, Eric Ricardo Carreon Ruiz"
copyright = "%Y, Peter Laszcz, Scott Young, Eric Ricardo Carreon Ruiz"

try:
    release = _version("sniff-core")
except PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Qt and NCrystal are imported at module scope. They are installed for real in
# CI so signatures stay accurate; list them here if a build environment cannot
# provide them.
autodoc_mock_imports: list[str] = []

# -- Napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Intersphinx -------------------------------------------------------------

# Fail fast instead of hanging the build when an inventory host is slow.
intersphinx_timeout = 15

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"
