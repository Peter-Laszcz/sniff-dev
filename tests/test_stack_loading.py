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
Building stacks from folders, file lists and arrays.
"""

import numpy as np
import pytest
import tifffile
from astropy.io import fits

from sni_app.core import (
    Stack,
    discover_and_load,
    discover_stack_dirs,
    list_stack_frames,
)

from conftest import (
    FRAME_SHAPE,
    SAMPLE_FRAMES,
    SAMPLE_STACK_DIR,
    make_frames,
    requires_sample_data,
    write_fits_folder,
)


@requires_sample_data
class TestSampleFolder:
    """Loading the sample acquisition."""

    def test_shape_and_headers(self, sample_stack):
        assert sample_stack.data.shape == (SAMPLE_FRAMES, *FRAME_SHAPE)
        assert sample_stack.data.dtype == np.float32
        assert len(sample_stack.headers) == SAMPLE_FRAMES

    def test_source_path_is_kept(self, sample_stack):
        assert sample_stack.path == SAMPLE_STACK_DIR

    def test_run_tables(self, sample_stack):
        run_meta = sample_stack.run_meta_data()
        assert sorted(run_meta) == ["shutter_count", "shutter_times", "spectra"]
        assert run_meta["spectra"].shape == (SAMPLE_FRAMES, 2)
        assert run_meta["shutter_count"].shape == (256, 2)
        assert run_meta["shutter_times"].shape == (256, 3)

    def test_tof(self, sample_stack):
        tofs = [float(header["TOF"]) for header in sample_stack.headers]
        assert tofs[0] == pytest.approx(0.00474)
        assert tofs == sorted(tofs)


class TestFromFolder:
    """Folder loading of frames."""

    def test_filename_order(self, tmp_path):
        frames = make_frames(5, seed=20)
        folder = write_fits_folder(tmp_path / "ordered", frames)

        stack = Stack.from_folder(folder)

        assert np.allclose(stack.data, frames, rtol=1e-6)

    def test_frame_mismatch_skip(self, tmp_path):
        folder = write_fits_folder(tmp_path / "ragged", make_frames(4, seed=21))
        fits.writeto(folder / "frame9999.fits", np.zeros((3, 3), dtype=np.float32))

        stack = Stack.from_folder(folder)

        assert stack.data.shape == (4, 8, 10)

    def test_empty_folder(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No readable image frames"):
            Stack.from_folder(empty)

    def test_progress_callback(self, tmp_path):
        folder = write_fits_folder(tmp_path / "progress", make_frames(4, seed=22))
        seen = []

        Stack.from_folder(folder, progress_callback=lambda done, total: seen.append((done, total)))

        assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_wavelength_headers_to_metadata(self, tmp_path):
        """A folder written by SNIFF carries its wavelengths in the headers."""
        folder = tmp_path / "wavelengths"
        folder.mkdir()
        for index in range(4):
            fits.writeto(
                folder / f"frame{index}.fits",
                np.full((4, 4), index + 1.0, dtype=np.float32),
                fits.Header({"wlength": 1.5 + index}),
            )

        stack = Stack.from_folder(folder)

        assert stack.stack_meta["wavelengths"] == [1.5, 2.5, 3.5, 4.5]


class TestFromFitsList:

    def test_reading_order(self, tmp_path):
        frames = make_frames(3, seed=23)
        folder = write_fits_folder(tmp_path / "list", frames)
        paths = list_stack_frames(folder)

        stack = Stack.from_fits_list(list(reversed(paths)))

        assert np.allclose(stack.data[0], frames[2], rtol=1e-6)

    def test_metadata_injection(self, tmp_path):
        folder = write_fits_folder(tmp_path / "meta", make_frames(2, seed=25))

        stack = Stack.from_fits_list(list_stack_frames(folder), meta={"display_name": "x"})

        assert stack.display_name() == "x"
        assert stack.path is None


class TestFromArray:
    """Array-backed stacks, including the shapes the GUI passes in."""

    def test_shape_retention(self):
        data = make_frames(3, seed=26)
        assert Stack.from_array(data).data.shape == (3, 8, 10)

    def test_2d_image_fits_3d(self):
        assert Stack.from_array(np.zeros((6, 7))).data.shape == (1, 6, 7)

    def test_profile_becomes_stack(self):
        assert Stack.from_array(np.arange(5.0)).data.shape == (5, 1, 1)

    def test_float32_cast(self):
        assert Stack.from_array(np.ones((2, 2, 2), dtype=np.int16)).data.dtype == np.float32

    def test_generate_absent_headers(self):
        assert len(Stack.from_array(make_frames(4, seed=27)).headers) == 4

    def test_header_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="Mismatch between number of headers"):
            Stack.from_array(make_frames(4, seed=28), headers=[fits.Header()])


class TestMultipageTiff:
    """Multipage TIFF loading."""

    def test_pages_to_frames(self, tmp_path):
        data = make_frames(4, seed=29)
        path = tmp_path / "stack.tif"
        tifffile.imwrite(path, data, photometric="minisblack")

        stack = Stack.from_multipage_tiff(path)

        assert stack.data.shape == data.shape
        assert np.allclose(stack.data, data)

    def test_stack_data_retention(self, tmp_path):
        data = make_frames(4, seed=38)
        Stack.from_array(data).save_stack("saved.tif", tmp_path)

        stack = Stack.from_multipage_tiff(tmp_path / "saved.tif")

        assert stack.data.shape == data.shape
        assert np.array_equal(stack.data, data)

    def test_header_provision(self, tmp_path):
        data = make_frames(3, seed=39)
        tifffile.imwrite(tmp_path / "stack.tif", data, photometric="minisblack")

        stack = Stack.from_multipage_tiff(tmp_path / "stack.tif")

        assert len(stack.headers) == 3

    def test_2d_fits_3d(self, tmp_path):
        path = tmp_path / "one.tif"
        tifffile.imwrite(path, make_frames(1, seed=40)[0], photometric="minisblack")

        assert Stack.from_multipage_tiff(path).data.shape == (1, 8, 10)

    def test_colour_conversion(self, tmp_path):
        colour = np.zeros((8, 10, 3), dtype=np.float32)
        colour[..., 0] = 3.0
        path = tmp_path / "colour.tif"
        tifffile.imwrite(path, colour, photometric="rgb")

        stack = Stack.from_multipage_tiff(path)

        assert stack.data.shape == (1, 8, 10)
        assert np.allclose(stack.data, 1.0)  # mean of (3, 0, 0)

    def test_metadata_injection(self, tmp_path):
        path = tmp_path / "stack.tif"
        tifffile.imwrite(path, make_frames(2, seed=41), photometric="minisblack")

        stack = Stack.from_multipage_tiff(path, meta={"display_name": "tiff stack"})

        assert stack.display_name() == "tiff stack"
        assert stack.path is None

    def test_shape_mismatch_raises_error(self, tmp_path):
        path = tmp_path / "profile.tif"
        tifffile.imwrite(path, np.arange(5, dtype=np.float32))

        with pytest.raises(ValueError, match="does not hold a frame series"):
            Stack.from_multipage_tiff(path)


class TestDiscovery:

    def test_valid_folder_discovery(self, tmp_path):
        write_fits_folder(tmp_path / "run_a", make_frames(2, seed=30))
        write_fits_folder(tmp_path / "run_b", make_frames(2, seed=31))
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "readme.txt").write_text("no images here")

        found = discover_stack_dirs(tmp_path)

        assert sorted(path.name for path in found) == sorted(
            ["run_a", "run_b", tmp_path.name]
        )

    def test_frames_are_listed_in_sorted_order(self, tmp_path):
        folder = write_fits_folder(tmp_path / "frames", make_frames(12, seed=32))
        names = [path.name for path in list_stack_frames(folder)]
        assert names == sorted(names)

    def test_exclude_unsupported_files(self, tmp_path):
        folder = write_fits_folder(tmp_path / "mixed", make_frames(2, seed=33))
        (folder / "log.txt").write_text("ignored")
        assert all(path.suffix == ".fits" for path in list_stack_frames(folder))

    def test_load_stack_with_weight(self, tmp_path):
        write_fits_folder(tmp_path / "sample_1", make_frames(3, seed=34))
        write_fits_folder(tmp_path / "ob_1", make_frames(3, seed=35))

        stacks, _params = discover_and_load(tmp_path)

        assert sorted(stack.path.name for stack in stacks) == ["ob_1", "sample_1"]
        weights = [stack.stack_meta["weights_data_frame"] for stack in stacks]
        assert all(frame is not None for frame in weights)
        assert list(weights[0].columns) == ["Folder", "w1", "w2", "OB1", "OB2"]

