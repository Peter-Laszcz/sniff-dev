"""
PyInstaller runtime hook to point NCrystal at the bundled shared library.
"""

import os
import sys

_lib = os.path.join(sys._MEIPASS, "NCrystal.dll")
if os.path.isfile(_lib):
    os.environ.setdefault("NCRYSTAL_LIB", _lib)
