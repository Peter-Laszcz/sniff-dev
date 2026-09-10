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
Region-of-interest cropping, profiles and the spectral analyses built on them.
"""

import numpy as np
import pytest
from scipy import constants

from sni_app.core import (
    BARNS_PER_CM2,
    Stack,
    atten_coefficient,
    clamp_roi_to_stack,
    compute_roi_stats,
    h_cross_section,
    relative_attenuation,
    roi_profile,
    roi_to_mask,
    roi_to_stack,
    stack_wavelengths,
    sum_of_logs_relative_attenuation,
    t_cross_section,
)

from conftest import make_stack, requires_sample_data


@pytest.fixture
def transmission_stack() -> Stack:
    """A 12-frame transmission stack with wavelengths."""
    rng = np.random.default_rng(200)
    data = (rng.random((12, 8, 10), dtype=np.float32) * 0.4 + 0.3).astype(np.float32)
    return Stack.from_array(
        data,
        stack_meta={"wavelengths": np.linspace(1.0, 6.0, 12)},
    )


class TestClamping:

    def test_valid_roi_retention(self, synthetic_stack):
        """i.e. roi inside frame is not altered"""
        assert clamp_roi_to_stack((1, 2, 3, 4), synthetic_stack) == (1, 2, 3, 4)

    def test_trim_oversized_roi(self, synthetic_stack):
        assert clamp_roi_to_stack((5, 5, 100, 100), synthetic_stack) == (5, 5, 5, 3)

    def test_over_negative_positions(self, synthetic_stack):
        assert clamp_roi_to_stack((-4, -7, 4, 4), synthetic_stack) == (0, 0, 4, 4)

    def test_over_positive_positions(self, synthetic_stack):
        assert clamp_roi_to_stack((50, 50, 4, 4), synthetic_stack) == (9, 7, 1, 1)


class TestProfiles:

    def test_mean_shape_match(self, synthetic_stack):
        profile = roi_profile((0, 0, 4, 4), synthetic_stack)
        assert profile.shape == (6,)
        assert profile.dtype == np.float64

    def test_profile_focuses_on_roi(self):
        data = np.zeros((2, 8, 10), dtype=np.float32)
        data[:, 0:2, 0:2] = 4.0
        stack = Stack.from_array(data)

        assert roi_profile((0, 0, 2, 2), stack) == pytest.approx([4.0, 4.0])


class TestRoiStats:

    def test_roi_stats(self):
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        stats = compute_roi_stats(frame, (0, 0, 2, 2))

        assert stats["mean"] == pytest.approx(2.5)  # 0, 1, 4, 5
        assert stats["median"] == pytest.approx(2.5)
        assert stats["min"] == 0.0
        assert stats["max"] == 5.0
        assert stats["valid_pixels"] == 4
        assert stats["invalid_pixels"] == 0

    def test_sem_approximates_std(self):
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        stats = compute_roi_stats(frame, (0, 0, 2, 2))
        assert stats["sem"] == pytest.approx(stats["std"] / np.sqrt(4))

    def test_invalid_pixel_detection(self):
        frame = np.ones((4, 4), dtype=np.float32)
        frame[0, 0] = np.nan
        frame[1, 1] = np.inf

        stats = compute_roi_stats(frame, (0, 0, 2, 2))

        assert stats["valid_pixels"] == 2
        assert stats["invalid_pixels"] == 2
        assert stats["mean"] == pytest.approx(1.0)

        frame = np.full((4, 4), np.nan, dtype=np.float32)
        stats = compute_roi_stats(frame, (0, 0, 2, 2))

        assert np.isnan(stats["mean"])
        assert stats["valid_pixels"] == 0

    def test_square_roi(self):
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert sorted(roi_to_mask(frame, (1, 1, 2, 2))) == [5.0, 6.0, 9.0, 10.0]

    def test_wide_roi(self):
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert sorted(roi_to_mask(frame, (0, 0, 3, 1))) == [0.0, 1.0, 2.0]

    def test_tall_roi(self):
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert sorted(roi_to_mask(frame, (0, 0, 1, 3))) == [0.0, 4.0, 8.0]


class TestRoiToStack:
    """Cropping stacks to ROI."""

    def test_valid_crop_dimension(self, synthetic_stack):
        out = roi_to_stack([synthetic_stack], (1, 2, 3, 4))[0]
        assert out.data.shape == (6, 4, 3)

    def test_valid_crop_values(self, synthetic_stack):
        out = roi_to_stack([synthetic_stack], (1, 2, 3, 4))[0]
        assert np.allclose(out.data, synthetic_stack.data[:, 2:6, 1:4])

    def test_crop_in_stack_history(self, synthetic_stack):
        out = roi_to_stack([synthetic_stack], (1, 2, 3, 4))[0]
        assert out.stack_meta["roi_crop_xywh"] == [1, 2, 3, 4]
        assert out.get_history()["params"] == {"roi_xywh": [1, 2, 3, 4]}

    def test_frame_times_retention(self, synthetic_stack):
        out = roi_to_stack([synthetic_stack], (0, 0, 2, 2))[0]
        assert np.allclose(out.times_of_flight(), synthetic_stack.times_of_flight())


class TestRelativeAttenuation:
    """Relative attenuation functionality."""

    def test_the_result_is_one_image(self, transmission_stack):
        out = relative_attenuation(transmission_stack, (0, 4), (6, 12))[0]
        assert out.data.shape == (1, 8, 10)

    def test_wavelength_bands(self, transmission_stack):
        out = relative_attenuation(transmission_stack, (0, 4), (6, 12))[0]
        assert out.get_history()["params"] == {
            "sw_range": [0, 4],
            "lw_range": [6, 12],
            "eps": 1e-6,
        }

    def test_analysis_results(self, transmission_stack):
        out = relative_attenuation(transmission_stack, (0, 4), (6, 12))[0]
        results = out.analysis_results()

        assert {"valid_count", "invalid_count", "den_median", "stack_mean"} <= set(results)
        assert results["valid_count"] + results["invalid_count"] == 80

    def test_sum_of_logs(self, transmission_stack):
        out = sum_of_logs_relative_attenuation(transmission_stack, (0, 4), (6, 12))[0]
        assert out.data.shape == (1, 8, 10)
        assert out.analysis_results()

    def test_sum_of_logs_binning(self, transmission_stack):
        out = sum_of_logs_relative_attenuation(
            transmission_stack, (0, 4), (6, 12), bin_factor=2
        )[0]
        assert out.data.shape == (1, 4, 5)


class TestAttenuationCoefficient:

    def test_spectrum_shape_match(self, transmission_stack):
        out = atten_coefficient(transmission_stack, transmission_stack)[0]
        assert out.data.shape == (12, 1, 1)

    def test_identical_stacks_cancel(self, transmission_stack):
        out = atten_coefficient(transmission_stack, transmission_stack)[0]
        assert np.allclose(out.data, 0.0, atol=1e-6)

    def test_log_ratio(self):
        sample = Stack.from_array(np.full((3, 4, 4), 0.5, dtype=np.float32))
        holder = Stack.from_array(np.full((3, 4, 4), 1.0, dtype=np.float32))

        out = atten_coefficient(sample, holder, d_cm=2.0)[0]

        assert np.allclose(out.data, -np.log(0.5) / 2.0, rtol=1e-5)

    def test_stack_shape_mismatch_raises_error(self, transmission_stack):
        with pytest.raises(ValueError, match="does not match stack Z length"):
            atten_coefficient(transmission_stack, make_stack(n_frames=3, seed=203))


    def test_non_positive_frames(self):
        sample = Stack.from_array(np.zeros((2, 4, 4), dtype=np.float32))
        holder = Stack.from_array(np.ones((2, 4, 4), dtype=np.float32))

        out = atten_coefficient(sample, holder)[0]

        assert np.all(np.isnan(out.data))


class TestTotalCrossSection:
    """Total microscopic cross-section spectra."""

    def test_spectrum_shape_match(self, transmission_stack):
        out = t_cross_section(transmission_stack, molar_mass=18.0, density=1.0)[0]
        assert out.data.shape == (12, 1, 1)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"molar_mass": 0.0, "density": 1.0}, "molar mass must be > 0"),
            ({"molar_mass": 18.0, "density": 0.0}, "density must be > 0"),
            ({"molar_mass": 18.0, "density": 1.0, "d_cm": 0.0}, "d must be > 0"),
        ],
    )
    def test_infeasible_parameters_raise_errors(self, transmission_stack, kwargs, message):
        with pytest.raises(ValueError, match=message):
            t_cross_section(transmission_stack, **kwargs)


class TestHydrogenCrossSection:
    """Hydrogen cross section of a compound mixture."""

    def test_spectrum_shape_match(self, transmission_stack):
        holder = Stack.from_array(
            np.ones_like(transmission_stack.data),
            stack_meta={"wavelengths": np.linspace(1.0, 6.0, 12)},
        )

        out = h_cross_section(transmission_stack, holder, ["C3H4O3"], [1.32])[0]

        assert out.data.shape == (12, 1, 1)

    def test_mixture_shape_mismatch_raises_error(self, transmission_stack):
        with pytest.raises(ValueError, match="must be the same length"):
            h_cross_section(
                transmission_stack, transmission_stack, ["C3H4O3", "C4H6O3"], [1.32]
            )

    def test_bad_formula_raises_error(self, transmission_stack):
        with pytest.raises(ValueError):
            h_cross_section(transmission_stack, transmission_stack, ["not a formula"], [1.0])


class TestStackWavelengths:
    """The per-frame wavelength axis analyses are plotted against."""

    def test_wavelength_import(self, transmission_stack):
        assert stack_wavelengths(transmission_stack).size == 12

    def test_wavelength_derivation(self):
        stack = make_stack(seed=204)
        stack.stack_meta["collimation_distance"] = 50.0
        stack.wavelengths(collimation_distance=50.0)  # records them on the stack
        assert stack_wavelengths(stack).size == 6

    def test_no_derivable_wavelengths_raises_error(self):
        with pytest.raises(ValueError, match="carries no per-frame wavelengths"):
            stack_wavelengths(make_stack(times=False, seed=205))

    def test_wavelength_shape_mismatch_raises_error(self):
        stack = make_stack(n_frames=6, seed=206)
        stack.stack_meta["wavelengths"] = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="only 2 wavelength"):
            stack_wavelengths(stack)

    @requires_sample_data
    def test_wavelength_shape_match(self, sample_stack):
        sample_stack.wavelengths(collimation_distance=56.4)
        assert stack_wavelengths(sample_stack).size == sample_stack.data.shape[0]
