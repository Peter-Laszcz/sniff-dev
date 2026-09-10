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
Stack and project I/O tests
"""

import numpy as np
import pandas as pd
import pytest
import tifffile

from sni_app.core import (
    Stack,
    export_stacks,
    list_stack_frames,
    load_project,
    save_project,
    stack_slice_acquisitions,
)

from conftest import make_stack, requires_sample_data


class TestSaveStack:
    """Writing one stack out as images."""

    def test_stack_data_export_roundtrip(self, synthetic_stack, tmp_path):
        synthetic_stack.save_stack("frames.fits", tmp_path)

        reloaded = Stack.from_fits_list(list_stack_frames(tmp_path), sort=True)

        assert np.allclose(reloaded.data, synthetic_stack.data, rtol=1e-6)

    def test_header_wavelength_roundtrip(self, synthetic_stack, tmp_path):
        synthetic_stack.stack_meta["wavelengths"] = np.linspace(1.0, 6.0, 6)

        synthetic_stack.save_stack("frames.fits", tmp_path)
        reloaded = Stack.from_folder(tmp_path)

        assert np.allclose(reloaded.stack_meta["wavelengths"], np.linspace(1.0, 6.0, 6))

    def test_tiff_exports_as_multipage(self, synthetic_stack, tmp_path):
        synthetic_stack.save_stack("frames.tif", tmp_path)

        assert (tmp_path / "frames.tif").exists()
        assert tifffile.imread(tmp_path / "frames.tif").shape == (6, 8, 10)

    def test_export_overwrite_protection(self, synthetic_stack, tmp_path):
        synthetic_stack.save_stack("frames.tif", tmp_path)
        with pytest.raises(FileExistsError):
            synthetic_stack.save_stack("frames.tif", tmp_path)

    def test_invalid_output_extension(self, synthetic_stack, tmp_path):
        with pytest.raises(ValueError, match="Cannot write stack to extension"):
            synthetic_stack.save_stack("frames.png", tmp_path)

    def test_export_unknown_extension(self, synthetic_stack, tmp_path):
        with pytest.raises(ValueError, match="Unrecognized file extension"):
            synthetic_stack.save_stack("frames.xyz", tmp_path)


class TestExportStacks:
    """Tests specific to batch exports."""

    def test_stack_name_preservation(self, tmp_path):
        pairs = [("sample A", make_stack(n_frames=2, seed=400)), ("sample B", make_stack(n_frames=2, seed=401))]

        written, errors = export_stacks(pairs, "run", ".fits", tmp_path)

        assert (written, errors) == (2, [])
        assert len(list(tmp_path.glob("run_sample*A_*.fits"))) == 2
        assert len(list(tmp_path.glob("run_sample*B_*.fits"))) == 2


    def test_overwrite_protection(self, tmp_path):
        """Writing without overwrite skips a frame that is already there."""
        export_stacks([("only", make_stack(n_frames=1, seed=405))], "run", ".fits", tmp_path)
        first = (tmp_path / "run_0000.fits").read_bytes()

        export_stacks([("only", make_stack(n_frames=1, seed=406))], "run", ".fits", tmp_path)

        assert (tmp_path / "run_0000.fits").read_bytes() == first

    def test_export_progress_callback(self, tmp_path):
        seen = []
        export_stacks(
            [("a", make_stack(n_frames=1, seed=405)), ("b", make_stack(n_frames=1, seed=406))],
            "run",
            ".fits",
            tmp_path,
            progress_callback=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(1, 2), (2, 2)]


class TestProjectFileRoundTrips:
    """Saving and loading whole projects."""

    def test_the_suffix_is_added_when_absent(self, synthetic_stack, tmp_path):
        path = save_project(tmp_path / "session", [synthetic_stack], None)
        assert path.name == "session.sniff"

    def test_stack_data_round_trip(self, synthetic_stack, tmp_path):
        path = save_project(tmp_path / "session.sniff", [synthetic_stack], None)

        project = load_project(path, lazy=False)
        try:
            assert len(project.stacks) == 1
            assert np.allclose(project.stacks[0].data, synthetic_stack.data)
        finally:
            project.close()

    def test_headers_round_trip(self, synthetic_stack, tmp_path):
        path = save_project(tmp_path / "session.sniff", [synthetic_stack], None)

        project = load_project(path, lazy=False)
        try:
            headers = project.stacks[0].headers
            assert len(headers) == 6
            assert float(headers[1]["TOF"]) == pytest.approx(0.001)
        finally:
            project.close()

    def test_frame_times_round_trip(self, synthetic_stack, tmp_path):
        path = save_project(tmp_path / "session.sniff", [synthetic_stack], None)

        project = load_project(path, lazy=False)
        try:
            assert np.allclose(
                project.stacks[0].times_of_flight(), synthetic_stack.times_of_flight()
            )
        finally:
            project.close()

    def test_stack_history_round_trip(self, tmp_path):
        entry = make_stack(seed=407)
        sliced = stack_slice_acquisitions([entry], start=0, stop=3)[0]
        path = save_project(tmp_path / "session.sniff", [entry, sliced], None)

        project = load_project(path, lazy=False)
        try:
            reloaded = {stack.stack_uuid(): stack for stack in project.stacks}
            assert reloaded[sliced.stack_uuid()].get_history()["process"] == "Stack Slicer"
            assert reloaded[sliced.stack_uuid()].parent_ids() == [entry.stack_uuid()]
        finally:
            project.close()

    def test_gui_state_round_trip(self, synthetic_stack, tmp_path):
        state = {"source_dir": "C:/runs", "function": "Normalisation", "window_half": 5}
        path = save_project(tmp_path / "session.sniff", [synthetic_stack], state)

        project = load_project(path, lazy=False)
        try:
            assert project.gui_state == state
        finally:
            project.close()

    def test_run_metadata_round_trip(self, synthetic_stack, tmp_path):
        synthetic_stack.stack_meta["run_meta"] = {
            "shutter_count": np.array([[0, 100], [1, 100]]),
        }
        path = save_project(tmp_path / "session.sniff", [synthetic_stack], None)

        project = load_project(path, lazy=False)
        try:
            counts = project.stacks[0].run_meta_data()["shutter_count"]
            assert np.array_equal(counts, [[0, 100], [1, 100]])
        finally:
            project.close()

    def test_weights_round_trip(self, synthetic_stack, tmp_path):
        synthetic_stack.stack_meta["weights_data_frame"] = pd.DataFrame(
            [{"Folder": "sample", "w1": 0.5, "w2": 0.5, "OB1": "ob_1", "OB2": "ob_2"}]
        )
        path = save_project(tmp_path / "session.sniff", [synthetic_stack], None)

        project = load_project(path, lazy=False)
        try:
            frame = project.stacks[0].stack_meta["weights_data_frame"]
            assert list(frame.columns) == ["Folder", "w1", "w2", "OB1", "OB2"]
            assert frame.loc[0, "OB1"] == "ob_1"
        finally:
            project.close()


    def test_invalid_project_rejected(self, tmp_path):
        import h5py

        path = tmp_path / "other.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["format"] = "SOMETHING_ELSE"

        with pytest.raises(ValueError, match="Not a SNIFF project file"):
            load_project(path)

    def test_stack_order_retention(self, tmp_path):
        stacks = [make_stack(n_frames=index + 1, seed=410 + index) for index in range(3)]
        path = save_project(tmp_path / "session.sniff", stacks, None)

        project = load_project(path, lazy=False)
        try:
            assert [stack.data.shape[0] for stack in project.stacks] == [1, 2, 3]
        finally:
            project.close()

    @requires_sample_data
    def test_stack_round_trips(self, sample_stack, tmp_path):
        path = save_project(tmp_path / "session.sniff", [sample_stack], None)

        project = load_project(path, lazy=False)
        try:
            reloaded = project.stacks[0]
            assert reloaded.data.shape == sample_stack.data.shape
            assert np.allclose(reloaded.data, sample_stack.data)
            assert sorted(reloaded.run_meta_data()) == [
                "shutter_count",
                "shutter_times",
                "spectra",
            ]
        finally:
            project.close()
