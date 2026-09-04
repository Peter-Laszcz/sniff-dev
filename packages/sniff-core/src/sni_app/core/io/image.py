# Spectroscopic Neutron Imaging Full-processing Framework (SNIFF)
# Copyright (C) 2026  ISIS Neutron and Muon Source
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code was primarily developed by Peter Laszcz.
# They can be contacted via laszczpeter@gmail.com.

"""
Single-image input/output.
Primarily supports TIFF and FITS.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from astropy.io import fits

from sni_app.core.process.img_processes import _frame_to_2d_float32, _squeeze_to_2d

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".fits"}
"""Every image suffix a stack can be built from."""

_TIFF_EXTENSIONS = {".tif", ".tiff"}
"""Needs its own reader (tifffile)."""

_FITS_EXTENSIONS = {".fits"}
"""Preferred for experiment metadata."""


def _get_img(path: Path) -> tuple[np.ndarray, fits.Header]:
    """
    Load an image file with FITS header (populated, if derivable).
    API Note: images are flipped upon reading for FITS consistency.

    Parameters
    ----------
    path: Path
        Path to image.

    Returns
    -------
    tuple
        Format: (image, FITS header for image).
    """
    _, ext = os.path.splitext(path)
    header = fits.Header()

    if ext.lower() in _FITS_EXTENSIONS:
        with fits.open(str(path)) as hdu_list:
            frame = hdu_list[0].data
            header = hdu_list[0].header
    elif ext.lower() in _TIFF_EXTENSIONS:
        frame = tifffile.imread(str(path))
    else:  # .png, .jpeg need flipping to match FITS format
        frame = np.flip(
            np.asarray(iio.imread(path)), axis=0
        )
    frame = _frame_to_2d_float32(_squeeze_to_2d(np.asarray(frame)))

    return frame, header


def _get_imgs_parallel(paths: list[Path], n_threads: int = 4):
    """
    Load image files in parallel.

    Parameters
    ----------
    paths: list[Path]
        List of image file paths.
    n_threads : int
        Number of processing threads to use in parallel. Capped in practicality by device limitations,
        i.e. number of physical processors.

    Returns
    -------
    tuple
        Format: (image, Header).
    """
    workers = max(1, min(n_threads, os.cpu_count() or 1, len(paths)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(lambda p: _get_img(p), paths)


def _write_img(
    img_tuple,
    file_name,
    base_dir: Path,
    overwrite=False,
    ext: str = ".fits",
):
    """
    Save an image and header to an image file (default FITS).

    Parameters
    ----------
    img_tuple : tuple[np.ndarray, Fits.Header]
        Format: (image, header).
    file_name : str
        Destination filename.
    base_dir : Path
        Destination folder.
    overwrite : bool
        Overwrite if file exists.
    ext : str
        Image extension. Defaults to ".fits" if not given.

    Returns
    -------
    bool
        True if written successfully.
    """
    ext = ext.lower()
    img, header = img_tuple
    file_name = f"{file_name}{ext}"
    dst_path = Path.joinpath(base_dir, file_name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if not overwrite and os.path.exists(dst_path):
        return False  # skipped if already exists and overwrite is off
    if ext in _FITS_EXTENSIONS:
        hdu = fits.PrimaryHDU(img, header=header)
        hdu.writeto(dst_path, overwrite=overwrite)
    elif ext in _TIFF_EXTENSIONS:
        if img.ndim == 2:
            img = img[None, :, :]
        tifffile.imwrite(str(dst_path), img)
    elif ext in ALLOWED_EXTENSIONS:
        # flips back to correct orientation
        iio.imwrite(str(dst_path), np.flip(np.asarray(img), axis=0))
    else:
        raise ValueError(f"Unrecognized file extension: {ext}")
    return True
