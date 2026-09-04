"""Sphinx configuration for the SNIFF documentation."""

import sys
from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("sniff-core", "sniff-gui"):
    sys.path.insert(0, str(_ROOT / "packages" / _pkg / "src"))

project = "SNIFF"
author = "Peter Laszcz, Scott Young, Eric Ricardo Carreon Ruiz"
copyright = ""

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

autodoc_mock_imports: list[str] = []

# -- Napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Intersphinx -------------------------------------------------------------

intersphinx_timeout = 15

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

def _demote_inventory_fetch_failures() -> None:
    import logging

    class _InventoryFetchFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno >= logging.WARNING and not getattr(record, "type", ""):
                record.levelno = logging.INFO
                record.levelname = "INFO"
            return True

    logging.getLogger("sphinx.sphinx.ext.intersphinx").addFilter(_InventoryFetchFilter())


_demote_inventory_fetch_failures()

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_show_copyright = False
