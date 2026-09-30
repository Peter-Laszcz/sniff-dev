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
AFGA and utils.
"""

import numpy as np
import pandas as pd
import pytest

from sni_app.core import (
    SPEC_BUILDING_BLOCKS,
    WAVELENGTH_MAX_A,
    WAVELENGTH_MIN_A,
    default_log_file,
    energies,
    energy_grid,
    log_dir,
    process_compound,
    read_timestamps_table,
    setup_logger,
    txt_timestamps,
    wavelengths,
)
from sni_app.core.util.scrubbing import _weighting_func

from conftest import make_frames, write_fits_folder


class TestEnergyGrid:

    def test_grid_shape(self):
        grid = energy_grid(points=100, energy_min=0.001, energy_max=0.3)

        assert grid.size == 100
        assert grid[0] == pytest.approx(0.001)
        assert grid[-1] == pytest.approx(0.3)

    @pytest.mark.parametrize("bounds", [(0.0, 0.3), (0.001, 0.0), (-1.0, 0.3)])
    def test_infeasible_bounds(self, bounds):
        with pytest.raises(ValueError, match="Energy bounds must be positive"):
            energy_grid(points=10, energy_min=bounds[0], energy_max=bounds[1])



class TestSpecBuildingBlocks:

    def test_unique_blocks(self):
        assert len(set(SPEC_BUILDING_BLOCKS)) == len(SPEC_BUILDING_BLOCKS)


class TestProcessCompound:

    def test_cs_spectrum_functionality(self):
        wavelength, cross_section = process_compound(
            name="Polyethylene",
            formula="C2H4",
            spec="2xCH2",
            density=0.94,
            scaling_factor=1.0,
            temperature=300.0,
            points=20,
        )

        assert wavelength.shape == cross_section.shape == (20,)
        assert np.all(cross_section > 0)

    def test_spectrum_order(self):
        wavelength, _cross_section = process_compound(
            "Polyethylene", "C2H4", "2xCH2", 0.94, 1.0, 300.0, points=20
        )
        assert wavelength[0] > wavelength[-1]


    @pytest.mark.parametrize(
        "wavelength_min, wavelength_max",
        [(0.0, 5.0), (-1.0, 5.0), (5.0, 5.0), (6.0, 5.0)],
    )
    def test_infeasible_range(self, wavelength_min, wavelength_max):
        with pytest.raises(ValueError, match="positive and increasing"):
            process_compound(
                "Polyethylene",
                "C2H4",
                "2xCH2",
                0.94,
                1.0,
                300.0,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                points=10,
            )


class TestTimestamps:
    """The per-folder timestamp table scrubbing weights are built from."""

    def test_a_row_is_written_for_each_folder(self, tmp_path):
        write_fits_folder(tmp_path / "sample_1", make_frames(2, seed=500))
        write_fits_folder(tmp_path / "ob_1", make_frames(2, seed=501))

        txt_timestamps(tmp_path, tmp_path)

        table = pd.read_csv(tmp_path / "timestamps.txt")
        assert set(table["Folder"]) >= {"sample_1", "ob_1"}

    def test_timestamp_record(self, tmp_path):
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=502))

        txt_timestamps(tmp_path, tmp_path)

        table = pd.read_csv(tmp_path / "timestamps.txt")
        assert list(table.columns) == [
            "Folder",
            "Modification (s)",
            "Creation (s)",
            "Formatted modification",
            "Formatted creation",
        ]
        assert table["Modification (s)"].gt(0).all()

    def test_no_blank_lines_between_records(self, tmp_path):
        """The writer ends its own lines; the text layer must not do it again."""
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=520))

        txt_timestamps(tmp_path, tmp_path)

        lines = (tmp_path / "timestamps.txt").read_text().splitlines()
        assert "" not in lines

    def test_header_written_into_an_empty_file(self, tmp_path):
        """A file left empty by an interrupted run is still given its header."""
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=521))
        (tmp_path / "timestamps.txt").touch()

        txt_timestamps(tmp_path, tmp_path)

        assert "sample_1" in set(read_timestamps_table(tmp_path)["Folder"])


class TestReadTimestampsTable:
    """Reading a timestamps table, whoever wrote it."""

    ROWS = [("ob_1", 100.0), ("sample_1", 150.0), ("ob_2", 200.0)]

    def write(self, path, header, sep):
        """Write ROWS under header, separated by sep, and return path."""
        lines = [sep.join(header)]
        lines += [sep.join([folder, str(when)]) for folder, when in self.ROWS]
        path.write_text("\n".join(lines) + "\n", newline="")
        return path

    @pytest.mark.parametrize("sep", [",", "\t", ";", "  "])
    def test_separators(self, tmp_path, sep):
        table = self.write(
            tmp_path / "timestamps.txt", ["Folder", "Modification (s)"], sep
        )

        frame = read_timestamps_table(table)

        assert list(frame["Folder"]) == [folder for folder, _ in self.ROWS]
        assert list(frame["Modification (s)"]) == [when for _, when in self.ROWS]

    @pytest.mark.parametrize(
        "header",
        [
            [" folder ", " Modification Time "],
            ["FOLDER", "mtime"],
            ["Directory", "Modified"],
        ],
    )
    def test_column_naming(self, tmp_path, header):
        table = self.write(tmp_path / "timestamps.txt", header, ",")

        assert list(read_timestamps_table(table)["Folder"]) == [
            folder for folder, _ in self.ROWS
        ]

    def test_folder_stands_in_for_the_table_it_holds(self, tmp_path):
        self.write(tmp_path / "timestamps.txt", ["Folder", "Modification (s)"], ",")

        assert list(read_timestamps_table(tmp_path)["Folder"]) == [
            folder for folder, _ in self.ROWS
        ]

    def test_rows_are_ordered_by_time(self, tmp_path):
        table = self.write(
            tmp_path / "timestamps.txt", ["Folder", "Modification (s)"], ","
        )

        assert list(read_timestamps_table(table)["Modification (s)"]) == sorted(
            when for _, when in self.ROWS
        )

    def test_repeated_folder_keeps_the_newest(self, tmp_path):
        """txt_timestamps appends, so a table built twice holds a folder twice."""
        table = tmp_path / "timestamps.txt"
        table.write_text(
            "\n".join(["Folder,Modification (s)", "sample_1,100", "sample_1,300"]),
            newline="",
        )

        frame = read_timestamps_table(table)

        assert list(frame["Folder"]) == ["sample_1"]
        assert list(frame["Modification (s)"]) == [300]

    def test_folder_names_are_not_read_as_blanks(self, tmp_path):
        table = tmp_path / "timestamps.txt"
        table.write_text(
            "\n".join(["Folder,Modification (s)", "NA,100", "null,200"]),
            newline="",
        )

        assert list(read_timestamps_table(table)["Folder"]) == ["NA", "null"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No timestamps table"):
            read_timestamps_table(tmp_path / "timestamps.txt")

    def test_unusable_table(self, tmp_path):
        table = tmp_path / "timestamps.txt"
        table.write_text("\n".join(["alpha,beta", "1,2"]), newline="")

        with pytest.raises(ValueError, match="Modification"):
            read_timestamps_table(table)

    def test_generated_table_reads_back(self, tmp_path):
        """What txt_timestamps writes is what read_timestamps_table expects."""
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=522))
        write_fits_folder(tmp_path / "ob_1", make_frames(1, seed=523))

        txt_timestamps(tmp_path, tmp_path)

        assert set(read_timestamps_table(tmp_path)["Folder"]) >= {"sample_1", "ob_1"}


class TestWeights:
    """Scrubbing Weights"""

    def test_weighting(self, tmp_path):
        write_fits_folder(tmp_path / "ob_1", make_frames(1, seed=503))
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=504))
        write_fits_folder(tmp_path / "ob_2", make_frames(1, seed=505))

        frame = _weighting_func(tmp_path)

        assert list(frame.columns) == ["Folder", "w1", "w2", "OB1", "OB2"]
        row = frame[frame["Folder"] == "sample_1"].iloc[0]
        assert row["w1"] + row["w2"] == pytest.approx(1.0)
        assert {row["OB1"], row["OB2"]} <= {"ob_1", "ob_2", "."}

    def test_empty_table_for_no_open_beams(self, tmp_path):
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=506))

        frame = _weighting_func(tmp_path)

        assert frame.empty
        assert list(frame.columns) == ["Folder", "w1", "w2", "OB1", "OB2"]

    def test_timestamps_table_from_elsewhere(self, tmp_path):
        """A table kept in a folder of its own still builds the weights."""
        write_fits_folder(tmp_path / "ob_1", make_frames(1, seed=524))
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=525))
        table = tmp_path / "elsewhere" / "timestamps.txt"
        table.parent.mkdir()
        table.write_text(
            "\n".join(
                [
                    "Folder\tModification (s)",
                    "ob_1\t100",
                    "sample_1\t150",
                    "ob_2\t200",
                ]
            ),
            newline="",
        )

        frame = _weighting_func(tmp_path, timestamps=table)

        assert not (tmp_path / "timestamps.txt").exists()  # nothing written to the run
        row = frame[frame["Folder"] == "sample_1"].iloc[0]
        assert (row["OB1"], row["OB2"]) == ("ob_1", "ob_2")
        assert row["w1"] == pytest.approx(0.5)
        assert row["w2"] == pytest.approx(0.5)

    def test_open_beam_folder_pointer(self, tmp_path):
        write_fits_folder(tmp_path / "flat_1", make_frames(1, seed=507))
        write_fits_folder(tmp_path / "sample_1", make_frames(1, seed=508))

        frame = _weighting_func(tmp_path, ob_folders=[tmp_path / "flat_1"])

        assert not frame.empty
        assert set(frame["OB1"]) == {"flat_1"}


class TestLogging:

    def test_log_dir(self):
        assert log_dir().name
        assert default_log_file().suffix == ".log"
        assert default_log_file().parent == log_dir()

    def test_log_write(self, tmp_path):
        path = tmp_path / "sniff.log"

        logger = setup_logger(log_file=path)
        logger.warning("a message worth keeping")
        for handler in logger.handlers:
            handler.flush()

        assert "a message worth keeping" in path.read_text(encoding="utf-8")

    def test_log_reuse(self, tmp_path):
        first = setup_logger(log_file=tmp_path / "sniff.log")
        second = setup_logger(log_file=tmp_path / "sniff.log")

        assert first is second
        assert len(first.handlers) == len(second.handlers)
