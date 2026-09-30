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
The stack-level processing functions.
"""

import numpy as np
import pytest
from scipy import ndimage

from sni_app.core import (
    Stack,
    compute_shutter_indices,
    stack_avg,
    stack_bin_frames,
    stack_join,
    stack_normalisation,
    stack_overlap_correction,
    stack_referencing,
    stack_registration,
    stack_sbkg_correction,
    stack_scrubbing,
    stack_slice_acquisitions,
    stack_stitching,
    stack_sum,
)

from conftest import make_frames, make_stack, requires_sample_data, write_fits_folder


@pytest.fixture
def black_body_mask() -> Stack:
    """A mask marking four black bodies, one per corner of a 16x16 frame."""
    mask = np.zeros((16, 16), dtype=np.float32)
    for y in (2, 12):
        for x in (2, 12):
            mask[y : y + 2, x : x + 2] = 1.0
    return Stack.from_array(mask)


class TestReductions:
    """Averaging and summing whole stacks together."""

    def test_elementwise_average(self):
        first, second = make_stack(seed=50), make_stack(seed=51)
        out = stack_avg([first, second])[0]
        assert np.allclose(out.data, (first.data + second.data) / 2)


    def test_averaging_nan_exclusion(self):
        first, second = make_stack(seed=56), make_stack(seed=57)
        first.data[0, 0, 0] = np.nan
        out = stack_avg([first, second])[0]
        assert out.data[0, 0, 0] == pytest.approx(second.data[0, 0, 0])

    def test_elementwise_sum(self):
        first, second = make_stack(seed=58), make_stack(seed=59)
        out = stack_sum([first, second])[0]
        assert np.allclose(out.data, first.data + second.data)

    def test_empty_sum_raises_error(self):
        with pytest.raises(ValueError, match="No stacks to sum"):
            stack_sum([])

    def test_sum_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="differing shapes"):
            stack_sum([make_stack(seed=60), make_stack(n_frames=4, seed=61)])


class TestJoin:
    """Concatenating stacks frame-wise."""

    def test_frames_are_concatenated_in_order(self):
        first, second = make_stack(n_frames=3, seed=62), make_stack(n_frames=2, seed=63)
        out = stack_join([first, second])[0]

        assert out.data.shape == (5, 8, 10)
        assert np.allclose(out.data[:3], first.data)
        assert np.allclose(out.data[3:], second.data)

    def test_header_retention(self):
        out = stack_join([make_stack(n_frames=3, seed=64), make_stack(n_frames=2, seed=65)])[0]
        assert len(out.headers) == 5

    def test_times_retention_where_relevant(self):
        out = stack_join([make_stack(n_frames=3, seed=66), make_stack(n_frames=2, seed=67)])[0]
        assert out.times_of_flight().size == 5

    def test_times_are_dropped_where_lacking(self):
        out = stack_join(
            [make_stack(n_frames=3, seed=68), make_stack(n_frames=2, seed=69, times=False)]
        )[0]
        assert out.times_of_flight() is None

    def test_joining_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="No stacks to join"):
            stack_join([])

    def test_concatenation_shape_mistmatch_raises_error(self):
        odd = make_stack(shape=(4, 4), seed=70)
        with pytest.raises(ValueError, match="differing frame shapes"):
            stack_join([make_stack(seed=71), odd])


class TestSlicing:

    def test_data_retention(self, synthetic_stack):
        out = stack_slice_acquisitions([synthetic_stack], start=1, stop=4)[0]
        assert out.data.shape == (3, 8, 10)
        assert np.allclose(out.data, synthetic_stack.data[1:4])

    def test_meta_retention(self, synthetic_stack):
        times = synthetic_stack.times_of_flight()
        out = stack_slice_acquisitions([synthetic_stack], start=2, stop=5)[0]

        assert len(out.headers) == 3
        assert np.allclose(out.times_of_flight(), times[2:5])

    def test_open_ends(self, synthetic_stack):
        out = stack_slice_acquisitions([synthetic_stack], start=4)[0]
        assert out.data.shape[0] == 2


class TestBinning:
    """Frame binning and the high/low energy split."""

    def test_data_functionality(self, synthetic_stack):
        out = stack_bin_frames([synthetic_stack], bin_factor=2, he_le=None)[0]

        assert out.data.shape == (3, 8, 10)
        assert np.allclose(out.data[0], synthetic_stack.data[:2].mean(axis=0))

    def test_delayed_start(self, synthetic_stack):
        out = stack_bin_frames([synthetic_stack], bin_factor=2, start_img=2, he_le=None)[0]

        assert out.data.shape[0] == 2
        assert np.allclose(out.data[0], synthetic_stack.data[2:4].mean(axis=0))

    def test_leftover_frames_dropped(self, synthetic_stack):
        out = stack_bin_frames([synthetic_stack], bin_factor=4, he_le=None)[0]
        assert out.data.shape[0] == 1

    def test_oversized_bin_raises_error(self, synthetic_stack):
        with pytest.raises(ValueError, match="No complete bins"):
            stack_bin_frames([synthetic_stack], bin_factor=99, he_le=None)

    @pytest.mark.parametrize("bin_factor", [0, -1])
    def test_bin_factor_below_one_raises_error(self, synthetic_stack, bin_factor):
        with pytest.raises(ValueError, match="at least 1"):
            stack_bin_frames([synthetic_stack], bin_factor=bin_factor, he_le=None)

    def test_energy_split(self):
        stack = make_stack(n_frames=60, seed=74)
        out = stack_bin_frames([stack], he_le=(True, [15, 30], [30, 50]))[0]

        assert out.data.shape == (2, 8, 10)
        assert np.allclose(out.data[0], stack.data[15:30].mean(axis=0))
        assert np.allclose(out.data[1], stack.data[30:50].mean(axis=0))


class TestReferencing:

    def test_every_frame_is_divided(self, synthetic_stack):
        reference = np.full((8, 10), 2.0, dtype=np.float32)
        out = stack_referencing([synthetic_stack], reference)[0]
        assert np.allclose(out.data, synthetic_stack.data / 2.0)


class TestNormalisation:

    @staticmethod
    def wide_stack(seed: int, n_frames: int = 4) -> Stack:
        """A stack whose frames the normalisation window fits inside."""
        return make_stack(n_frames=n_frames, shape=(16, 16), seed=seed, scale=1000.0)

    def test_shape_retention(self):
        stack = self.wide_stack(75)
        out = stack_normalisation([stack], self.wide_stack(76))[0]

        assert out.data.shape == stack.data.shape
        assert out.data.dtype == np.float32

    def test_shape_mismatch_raises_error(self, synthetic_stack):
        with pytest.raises(ValueError, match="shape mismatch"):
            stack_normalisation([self.wide_stack(77)], self.wide_stack(78, n_frames=3))
        with pytest.raises(ValueError, match="too small for 11x11 window"):
            stack_normalisation([synthetic_stack], make_stack(seed=79))

    def test_scaling_factor(self):
        stack, open_beam = self.wide_stack(81), self.wide_stack(82)
        plain = stack_normalisation([stack], open_beam)[0]
        scaled = stack_normalisation([stack], open_beam, scale=2.0)[0]

        assert np.allclose(scaled.data, plain.data * 2.0, rtol=1e-5)

    def test_progress_callback(self):
        seen = []
        stack_normalisation(
            [self.wide_stack(83), self.wide_stack(84)],
            self.wide_stack(85),
            progress_callback=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(1, 2), (2, 2)]

    def test_history_recording(self):
        open_beam = self.wide_stack(86)
        out = stack_normalisation([self.wide_stack(87)], open_beam, window_half=3)[0]

        history = out.get_history()
        assert history["aux"] == {"open_beam": open_beam.robust_stack_uuid()}
        assert history["params"]["window_half"] == 3


class TestOverlapCorrection:

    def test_a_stack_without_run_tables_is_an_error(self, synthetic_stack):
        with pytest.raises(ValueError, match="No file selected"):
            stack_overlap_correction([synthetic_stack])

    @requires_sample_data
    def test_shape_retention(self, sample_stack):
        out = stack_overlap_correction([sample_stack])[0]

        assert out.data.shape == sample_stack.data.shape
        assert np.all(np.isfinite(out.data))
        assert len(out.headers) == sample_stack.data.shape[0]

    @requires_sample_data
    def test_correction_raises_the_counts_it_recovers(self, sample_stack):
        """Overlap correction divides by an occupancy below one, so counts rise."""
        out = stack_overlap_correction([sample_stack])[0]

        assert out.data.sum() > sample_stack.data.sum()
        assert np.allclose(out.data[0], sample_stack.data[0])  # first frame unoccupied

    @requires_sample_data
    def test_run_meta_retention(self, sample_stack):
        """The GUI reads shutter counts back off the corrected stack."""
        out = stack_overlap_correction([sample_stack])[0]
        assert sorted(out.run_meta_data()) == ["shutter_count", "shutter_times", "spectra"]

    @requires_sample_data
    def test_progress_is_reported_per_stack(self, sample_stack):
        seen = []
        stack_overlap_correction(
            [sample_stack], progress_callback=lambda done, total: seen.append((done, total))
        )
        assert seen == [(1, 1)]


class TestShutterIndices:
    """Mapping shutter intervals onto frame indices."""

    def test_shutter_to_frame_range(self):
        shutter_count = np.array([[0, 100], [1, 100]])
        shutter_times = np.array([[0, 0.0, 0.005], [1, 0.0, 0.005]])
        spectra = np.column_stack([np.arange(10) * 0.001, np.arange(10)])

        starts, ends, counts = compute_shutter_indices(shutter_count, shutter_times, spectra)

        assert starts.tolist() == [0, 5]
        assert ends.tolist() == [5, 10]
        assert counts.tolist() == [100, 100]

    def test_shutters_stop_at_the_first_empty_count(self):
        shutter_count = np.array([[0, 100], [1, 0], [2, 100]])
        shutter_times = np.array([[0, 0.0, 0.005], [1, 0.0, 0.005], [2, 0.0, 0.005]])
        spectra = np.column_stack([np.arange(10) * 0.001, np.arange(10)])

        starts, _ends, counts = compute_shutter_indices(shutter_count, shutter_times, spectra)

        assert starts.size == 1
        assert counts.tolist() == [100]

    @requires_sample_data
    def test_the_sample_run_covers_every_frame(self, sample_stack):
        run_meta = sample_stack.run_meta_data()
        starts, ends, counts = compute_shutter_indices(
            run_meta["shutter_count"], run_meta["shutter_times"], run_meta["spectra"]
        )

        assert starts[0] == 0
        assert ends[-1] == sample_stack.data.shape[0]
        assert np.all((counts >= 17939) & (counts <= 17941))  # one run's triggers


class TestStitching:

    def test_data_stitch(self):
        short = make_stack(n_frames=8, seed=83)
        long = make_stack(n_frames=8, seed=84)

        out = stack_stitching(short, long, (0, 4), (0, 4), 0.03683, 56.25)[0]

        assert out.data.shape[1:] == short.data.shape[1:]
        assert 0 < out.data.shape[0] <= 8

    def test_shape_mismatch_raises_error(self):
        short = make_stack(n_frames=4, seed=85)
        long = make_stack(n_frames=4, shape=(4, 4), seed=86)

        with pytest.raises(ValueError, match="different frame dimensions"):
            stack_stitching(short, long)

    def test_empty_range_raises_error(self):
        short, long = make_stack(seed=87), make_stack(seed=88)
        with pytest.raises(ValueError, match="ranges are empty"):
            stack_stitching(short, long, (0, 0), (0, 0))


class TestBlackBodyCorrection:
    """Spatial background (black body) correction."""

    def test_shape_retention(self, black_body_mask):
        stack = make_stack(n_frames=3, shape=(16, 16), seed=91)
        out = stack_sbkg_correction([stack], black_body_mask)[0]

        assert out.data.shape == stack.data.shape

    def test_3d_mask_raises_error(self):
        mask = Stack.from_array(np.ones((2, 16, 16), dtype=np.float32))
        stack = make_stack(shape=(16, 16), seed=92)
        with pytest.raises(ValueError, match="must be a single image"):
            stack_sbkg_correction([stack], mask)

    def test_sparse_black_bodies_raises_error(self):
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[1:3, 1:3] = 1.0
        stack = make_stack(shape=(16, 16), seed=93)
        with pytest.raises(ValueError, match="needs at least"):
            stack_sbkg_correction([stack], Stack.from_array(mask))

    def test_mask_shape_mismatch_raises_error(self, black_body_mask):
        with pytest.raises(ValueError, match="does not match frame shape"):
            stack_sbkg_correction([make_stack(seed=94)], black_body_mask)

    def test_progress_callback(self, black_body_mask):
        seen = []
        stack_sbkg_correction(
            [make_stack(n_frames=2, shape=(16, 16), seed=95)],
            black_body_mask,
            progress_callback=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(1, 2), (2, 2)]

    def test_mask_recorded_as_auxiliary_input(self, black_body_mask):
        stack = make_stack(n_frames=2, shape=(16, 16), seed=96)
        out = stack_sbkg_correction([stack], black_body_mask)[0]
        assert out.get_history()["aux"] == {"bb_mask": black_body_mask.robust_stack_uuid()}


class TestScrubbing:

    def test_lack_of_data_raises_error(self, synthetic_stack):
        with pytest.raises(ValueError, match="weights dataframe or an"):
            stack_scrubbing([synthetic_stack])

    def test_open_beam_divides_every_frame(self, tmp_path):
        open_beam_dir = write_fits_folder(
            tmp_path / "ob", np.full((3, 8, 10), 4.0, dtype=np.float32)
        )
        stack = make_stack(seed=97)
        expected = stack.data / 4.0

        out = stack_scrubbing([stack], open_beam_dir=open_beam_dir)[0]

        assert np.allclose(out.data, expected)
        assert out.get_history()["params"]["open_beam_dir"] == str(open_beam_dir)

    def test_skip_unweighted_stacks(self, tmp_path):
        import pandas as pd

        stack = Stack.from_folder(write_fits_folder(tmp_path / "sample", make_frames(2, seed=98)))
        weights = pd.DataFrame(columns=["Folder", "w1", "w2", "OB1", "OB2"])

        assert stack_scrubbing([stack], weights=weights) == []

    def test_open_beam_selection(self, tmp_path):
        import pandas as pd

        write_fits_folder(tmp_path / "ob_1", np.full((2, 8, 10), 2.0, dtype=np.float32))
        write_fits_folder(tmp_path / "ob_2", np.full((2, 8, 10), 4.0, dtype=np.float32))
        sample_dir = write_fits_folder(tmp_path / "sample", make_frames(2, seed=99))
        stack = Stack.from_folder(sample_dir)
        weights = pd.DataFrame(
            [{"Folder": "sample", "w1": 0.5, "w2": 0.5, "OB1": "ob_1", "OB2": "ob_2"}]
        )

        out = stack_scrubbing([stack], weights=weights)[0]

        assert np.allclose(out.data, stack.data / 3.0)  # 0.5*2 + 0.5*4


class TestRegistration:

    @staticmethod
    def textured_frame(seed: int = 100) -> np.ndarray:
        """A 64x64 frame of 8-pixel blocks, lightly speckled."""
        rng = np.random.default_rng(seed)
        frame = np.kron(
            rng.integers(0, 2, size=(8, 8)), np.ones((8, 8))
        ).astype(np.float32) * 100.0
        return frame + rng.random(frame.shape, dtype=np.float32) * 5.0

    def test_translation(self):
        base = self.textured_frame()
        shifted = ndimage.shift(base, (1.0, -1.0), order=1, mode="nearest")
        stack = Stack.from_array(np.stack([base, shifted]).astype(np.float32))

        out = stack_registration([stack], Stack.from_array(base), max_workers=1)[0]

        assert out.data.shape == stack.data.shape
        assert np.all(np.isfinite(out.data))
        assert np.abs(out.data[1] - base).mean() < np.abs(shifted - base).mean() / 2

    def test_record_registration_failures(self):
        base = self.textured_frame()
        stack = Stack.from_array(
            np.stack([base, np.zeros_like(base)]).astype(np.float32)
        )

        out = stack_registration([stack], Stack.from_array(base), max_workers=1)[0]

        assert out.stack_meta["registration_skipped_frames"] == [1]
        assert np.allclose(out.data[1], 0.0)

    def test_reference_shape_mismatch_throws_error(self):
        stack = Stack.from_array(np.stack([self.textured_frame()]))
        reference = Stack(
            data=np.zeros((2, 2, 4, 4), dtype=np.float32),
            headers=[],
            stack_meta={},
            path=None,
        )
        with pytest.raises(ValueError, match="reduce to a 2D image"):
            stack_registration([stack], reference, max_workers=1)

