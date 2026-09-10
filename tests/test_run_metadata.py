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
The experiment run tables: shutter counts, shutter times and spectra.
"""

import numpy as np
import pytest

from sni_app.core import (
    extract_run_stats,
    frame_wavelengths,
    resolve_run_meta_array,
    scan_experiment_txts,
)
from sni_app.core.util.run_stats import _first_shutter_count

from conftest import (
    SAMPLE_FRAMES,
    SAMPLE_OPEN_BEAM_DIR,
    SAMPLE_STACK_DIR,
    requires_sample_data,
)


def write_run_tables(directory, rows: int = 4, count: int = 100):
    """Write a minimal set of run tables into a folder."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_ShutterCount.txt").write_text(
        "".join(f"{index}\t{count}\n" for index in range(rows))
    )
    (directory / "run_ShutterTimes.txt").write_text(
        "".join(f"{index}\t0.001\t0.01\n" for index in range(rows))
    )
    (directory / "run_Spectra.txt").write_text(
        "".join(f"{index * 0.001:.6f}\t{index}\n" for index in range(rows))
    )
    return directory


class TestScanExperimentTxts:

    def test_all_roles_found(self, tmp_path):
        found = scan_experiment_txts(write_run_tables(tmp_path / "run"))
        assert sorted(found) == ["shutter_count", "shutter_times", "spectra"]

    def test_shape_retention(self, tmp_path):
        found = scan_experiment_txts(write_run_tables(tmp_path / "run", rows=5))
        assert found["shutter_count"].shape == (5, 2)
        assert found["shutter_times"].shape == (5, 3)
        assert found["spectra"].shape == (5, 2)

    def test_tableless_data_yields_nothing(self, tmp_path):
        empty = tmp_path / "bare"
        empty.mkdir()
        assert scan_experiment_txts(empty) == {}


    def test_only_present_roles_returned(self, tmp_path):
        partial = tmp_path / "partial"
        partial.mkdir()
        (partial / "run_Spectra.txt").write_text("0.0\t0\n0.001\t1\n")
        assert sorted(scan_experiment_txts(partial)) == ["spectra"]

    @requires_sample_data
    def test_samples_return_tables(self):
        for directory in (SAMPLE_STACK_DIR, SAMPLE_OPEN_BEAM_DIR):
            found = scan_experiment_txts(directory)
            assert sorted(found) == ["shutter_count", "shutter_times", "spectra"]
            assert found["spectra"].shape == (SAMPLE_FRAMES, 2)


class TestResolveRunMetaArray:
    """Choosing between a stack's own tables and an override file."""

    def test_internal_data_is_used_by_default(self, tmp_path):
        run_meta = scan_experiment_txts(write_run_tables(tmp_path / "run"))
        resolved = resolve_run_meta_array({"run_meta": run_meta}, "spectra")
        assert np.array_equal(resolved, run_meta["spectra"])

    def test_internal_data_override(self, tmp_path):
        folder = write_run_tables(tmp_path / "run")
        override = tmp_path / "other_Spectra.txt"
        override.write_text("9.0\t9\n")

        resolved = resolve_run_meta_array(
            {"run_meta": scan_experiment_txts(folder)}, "spectra", str(override)
        )

        assert resolved.tolist() == [[9.0, 9]]

    def test_nothing_to_resolve_raises_error(self):
        with pytest.raises(ValueError, match="No file selected"):
            resolve_run_meta_array(None, "spectra")

    def test_missing_role_raises_error(self, tmp_path):
        run_meta = scan_experiment_txts(write_run_tables(tmp_path / "run"))
        del run_meta["shutter_count"]
        with pytest.raises(ValueError, match="no internal correction data"):
            resolve_run_meta_array({"run_meta": run_meta}, "shutter_count")


class TestExtractRunStats:

    def test_the_three_statistics_are_reported(self, tmp_path):
        stats = extract_run_stats(write_run_tables(tmp_path / "run", rows=4))
        assert sorted(stats) == ["Shutter Count", "Shutter Times", "Spectral Times"]

    def test_each_statistic_is_extracted(self, tmp_path):
        stats = extract_run_stats(write_run_tables(tmp_path / "run", rows=4))
        assert stats["Shutter Count"].tolist() == [100, 100, 100, 100]
        assert stats["Spectral Times"] == pytest.approx([0.0, 0.001, 0.002, 0.003])

    def test_absent_files_give_none(self, tmp_path):
        empty = tmp_path / "bare"
        empty.mkdir()
        stats = extract_run_stats(empty)
        assert set(stats.values()) == {None}

    def test_only_present_tables_read(self, tmp_path):
        partial = tmp_path / "partial"
        partial.mkdir()
        (partial / "run_Spectra.txt").write_text("0.0\t0\n0.001\t1\n")

        stats = extract_run_stats(partial)

        assert stats["Spectral Times"] == pytest.approx([0.0, 0.001])
        assert stats["Shutter Count"] is None

    @requires_sample_data
    def test_sample_returns_shutter_counts(self):
        stats = extract_run_stats(SAMPLE_STACK_DIR)

        assert stats["Shutter Count"][0] == 17941
        assert stats["Spectral Times"].size == SAMPLE_FRAMES


class NormalisationShutterCount:

    def test_no_tables_means_no_count(self):
        assert _first_shutter_count(None) is None
        assert _first_shutter_count({}) is None

    @requires_sample_data
    def test_sample_returns_correct_scale(
        self, sample_stack, open_beam_stack
    ):
        sample_count = _first_shutter_count(sample_stack.run_meta_data())
        open_beam_count = _first_shutter_count(open_beam_stack.run_meta_data())

        assert sample_count == pytest.approx(17941.0)
        assert open_beam_count == pytest.approx(26535.0)
        assert open_beam_count / sample_count == pytest.approx(1.4790, abs=1e-4)


class TestFrameWavelengths:

    def test_scaling_functionality(self):
        times = np.array([0.01, 0.02])
        assert frame_wavelengths(times, 0.0, 50.0) == pytest.approx(
            [3956.0 / 50.0 * 0.01, 3956.0 / 50.0 * 0.02]
        )

    def test_delay_functionality(self):
        times = np.array([0.01])
        delayed = frame_wavelengths(times, 0.005, 50.0, apply_delay=True)
        assert delayed == pytest.approx([3956.0 / 50.0 * 0.015])
