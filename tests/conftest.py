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
Shared fixtures for the core-library test suite.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from astropy.io import fits

from sni_app.core import Stack

SAMPLE_DATA = Path(__file__).parent / "sample_data"
"""Root of the sample folders."""

SAMPLE_STACK_DIR = SAMPLE_DATA / "stack_uncorrected"
"""Experimental sample stack."""

SAMPLE_OPEN_BEAM_DIR = SAMPLE_DATA / "open_beam_overlap_corrected"
"""Sample open beam stack."""

FRAME_SHAPE = (64, 64)
"""Frame shape of both sample folders."""

SAMPLE_FRAMES = 473
"""Frames each sample folder holds after dropping summed image."""


def sample_data_missing(*dirs: Path) -> bool:
    return not all(directory.is_dir() and any(directory.glob("*.fits")) for directory in dirs)


requires_sample_data = pytest.mark.skipif(
    sample_data_missing(SAMPLE_STACK_DIR, SAMPLE_OPEN_BEAM_DIR),
    reason="tests/sample_data is not populated",
)
"""Skip mark for the tests using sample data"""


def clone(stack: Stack) -> Stack:
    """
    Deep stack copy

    Parameters
    ----------
    stack : Stack
        Stack to copy.

    Returns
    -------
    Stack
    """
    return Stack(
        data=np.array(stack.data, copy=True),
        headers=[header.copy() for header in stack.headers],
        stack_meta={
            key: value for key, value in (stack.stack_meta or {}).items()
        },
        path=stack.path,
    )


def make_frames(
    n_frames: int = 6,
    shape: Tuple[int, int] = (8, 10),
    seed: int = 0,
    scale: float = 100.0,
) -> np.ndarray:
    """
    Build a reproducible float32 frame series.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    shape : tuple[int, int]
        Frame shape (height, width).
    seed : int
        Seed of the generator.
    scale : float
        Upper bound of the pixel values.

    Returns
    -------
    np.ndarray
        Frames of shape (n_frames, *shape), as float32.
    """
    rng = np.random.default_rng(seed)
    return (rng.random((n_frames, *shape), dtype=np.float32) * scale).astype(np.float32)


def make_stack(
    n_frames: int = 6,
    shape: Tuple[int, int] = (8, 10),
    seed: int = 0,
    scale: float = 100.0,
    times: bool = True,
) -> Stack:
    """
    Build a stack of reproducible frames, optionally carrying frame times.

    Parameters
    ----------
    n_frames, shape, seed, scale
        Passed to make_frames.
    times : bool
        Whether to record one time of flight per frame (0.001 s apart).

    Returns
    -------
    Stack
        The synthetic stack, with one header per frame.
    """
    data = make_frames(n_frames, shape, seed, scale)
    headers = [fits.Header({"TOF": 0.001 * index}) for index in range(n_frames)]
    meta: Dict = {}
    if times:
        meta["spectra_times"] = np.arange(n_frames, dtype=np.float64) * 0.001
    return Stack(data=data, headers=headers, stack_meta=meta, path=None)


def write_fits_folder(
    directory: Path,
    data: np.ndarray,
    stem: str = "frame",
    headers: bool = True,
) -> Path:
    """
    Write a frame series into a folder as individual FITS files.

    Parameters
    ----------
    directory : Path
        Folder to write into; created if absent.
    data : np.ndarray
        Frames to write (axis order TYX).
    stem : str
        Filename stem; files are numbered in frame order.
    headers : bool
        Whether to give each file a header containing TOF.

    Returns
    -------
    Path
        The folder written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(np.asarray(data)):
        header = fits.Header({"TOF": 0.001 * index}) if headers else None
        fits.writeto(
            directory / f"{stem}{index:04d}.fits",
            np.asarray(frame, dtype=np.float32),
            header,
            overwrite=True,
        )
    return directory


@pytest.fixture(scope="session")
def _sample_stacks() -> Dict[str, Stack]:
    """Read sample folders to check for data"""
    if sample_data_missing(SAMPLE_STACK_DIR, SAMPLE_OPEN_BEAM_DIR):
        pytest.skip("tests/sample_data is not populated")
    return {
        "sample": Stack.from_folder(SAMPLE_STACK_DIR),
        "open_beam": Stack.from_folder(SAMPLE_OPEN_BEAM_DIR),
    }


@pytest.fixture
def sample_stack(_sample_stacks: Dict[str, Stack]) -> Stack:
    """Copy of sample experimental stack."""
    return clone(_sample_stacks["sample"])


@pytest.fixture
def open_beam_stack(_sample_stacks: Dict[str, Stack]) -> Stack:
    """Copy of open-beam experimental stack."""
    return clone(_sample_stacks["open_beam"])


@pytest.fixture
def synthetic_stack() -> Stack:
    """Small reproducible stack carrying frame times."""
    return make_stack()
