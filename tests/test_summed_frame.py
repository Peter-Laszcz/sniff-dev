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
Summed image detection
"""

import numpy as np
import pytest

from sni_app.core import Stack, summed_frame_index
from sni_app.core.components.stack import _spectra_implied_summed_frame

from conftest import (
    SAMPLE_FRAMES,
    SAMPLE_STACK_DIR,
    make_frames,
    requires_sample_data,
    write_fits_folder,
)


def with_sum_at(frames: np.ndarray, index: int) -> np.ndarray:
    """Insert the sum of the given frames at index."""
    return np.insert(frames, index, frames.sum(axis=0), axis=0).astype(np.float32)


@pytest.mark.parametrize("index", [0, 3, 6])
def test_summed_image(index):
    frames = make_frames(6, seed=1)
    assert summed_frame_index(with_sum_at(frames, index)) == index


def test_no_summed_image():
    assert summed_frame_index(make_frames(6, seed=2)) is None



def test_rounding_tolerance():
    frames = make_frames(6, seed=5)
    summed = frames.sum(axis=0) * (1 + 1e-6)
    assert summed_frame_index(np.concatenate([frames, summed[None]])) == 6

def test_empty_stack():
    assert summed_frame_index(np.zeros((5, 4, 4), dtype=np.float32)) is None


class TestFromFolder:
    """The loader drops a summed image it can recognise."""

    def test_summed_frame_dropped_without_spectra_file(self, tmp_path):
        """The gap the numeric test fills: a folder holding no run tables."""
        frames = make_frames(5, seed=9)
        folder = write_fits_folder(tmp_path / "summed", with_sum_at(frames, 5))

        stack = Stack.from_folder(folder)

        assert stack.data.shape == (5, *frames.shape[1:])
        assert len(stack.headers) == 5
        assert np.allclose(stack.data, frames, rtol=1e-5)

    def test_plain_folder_keeps_every_frame(self, tmp_path):
        frames = make_frames(5, seed=11)
        folder = write_fits_folder(tmp_path / "plain", frames)

        stack = Stack.from_folder(folder)

        assert stack.data.shape[0] == 5
        assert len(stack.headers) == 5

    def test_headers_frame_match_retention(self, tmp_path):
        """Header 0 must still describe frame 0 after a leading drop."""
        frames = make_frames(4, seed=12)
        folder = write_fits_folder(tmp_path / "headers", with_sum_at(frames, 0))

        stack = Stack.from_folder(folder)

        tofs = [float(header["TOF"]) for header in stack.headers]
        assert tofs == pytest.approx([0.001, 0.002, 0.003, 0.004])

    @requires_sample_data
    def test_sample_drops_summed_image(self, sample_stack):
        assert len(list(SAMPLE_STACK_DIR.glob("*.fits"))) == SAMPLE_FRAMES + 1
        assert sample_stack.data.shape[0] == SAMPLE_FRAMES
        assert len(sample_stack.headers) == SAMPLE_FRAMES


class TestSpectraFallback:
    """Fallback where overflow prevents image summation detection"""

    def test_functionality(self):
        run_meta = {"spectra": np.zeros((10, 2))}
        assert _spectra_implied_summed_frame(11, run_meta) == 10

    def test_shape_match_returns_nothing(self):
        run_meta = {"spectra": np.zeros((10, 2))}
        assert _spectra_implied_summed_frame(10, run_meta) is None

    def test_no_spectra_table_returns_nothing(self):
        assert _spectra_implied_summed_frame(11, None) is None
        assert _spectra_implied_summed_frame(11, {}) is None
