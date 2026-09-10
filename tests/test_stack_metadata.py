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
A stack's identity, provenance, frame times and analysis results.
"""

import numpy as np
import pytest

from sni_app.core import (
    Stack,
    frame_wavelengths,
    record_derivation,
    stack_slice_acquisitions,
    stack_sum,
)

from conftest import make_stack, requires_sample_data


class TestIdentity:
    """Stack UUIDs and display names."""

    def test_initialised_stack_lacks_uuid(self):
        assert make_stack().stack_uuid() is None

    def test_uuid_assignment(self):
        stack = make_stack()
        assigned = stack.assign_stack_uuid()
        assert stack.stack_uuid() == assigned

    def test_robust_uuid_stability(self):
        stack = make_stack()
        assert stack.robust_stack_uuid() == stack.robust_stack_uuid()

    def test_uuid_reassignment(self):
        stack = make_stack()
        first = stack.robust_stack_uuid()
        assert stack.assign_stack_uuid() != first

    def test_display_name_fallback(self):
        assert make_stack().display_name() == "stack"

    def test_display_name_setter(self):
        stack = make_stack()
        stack.stack_meta["display_name"] = "sample A"
        assert stack.display_name() == "sample A"

    @requires_sample_data
    def test_stack_named_after_folder(self, sample_stack):
        assert sample_stack.display_name() == "stack_uncorrected"


class TestProvenance:
    """History recorded by the processes that produce stacks."""

    def test_fresh_stack_is_entry_point(self):
        stack = make_stack()
        assert stack.is_entry_point()
        assert stack.get_history() is None
        assert stack.parent_ids() == []

    def test_process_record(self, synthetic_stack):
        out = stack_slice_acquisitions([synthetic_stack], start=1, stop=4)[0]

        history = out.get_history()
        assert history["process"] == "Stack Slicer"
        assert history["params"] == {"start": 1, "stop": 4}
        assert history["mode"] == "map"

    def test_stack_derivation(self, synthetic_stack):
        parent_id = synthetic_stack.robust_stack_uuid()

        out = stack_slice_acquisitions([synthetic_stack], start=0, stop=2)[0]

        assert out.parent_ids() == [parent_id]
        assert not out.is_entry_point()
        assert out.stack_uuid() != parent_id

    def test_reduction_history(self):
        first, second = make_stack(seed=40), make_stack(seed=41)
        ids = [first.robust_stack_uuid(), second.robust_stack_uuid()]

        out = stack_sum([first, second])[0]

        assert out.parent_ids() == ids
        assert out.get_history()["mode"] == "reduce"

    def test_history_accumulation(self, synthetic_stack):
        sliced = stack_slice_acquisitions([synthetic_stack], start=0, stop=4)[0]
        summed = stack_sum([sliced])[0]

        chain = [step["process"] for step in summed.process_history()]
        assert chain == ["Stack Slicer", "Stack Summation"]

    def test_history_string(self, synthetic_stack):
        out = stack_slice_acquisitions([synthetic_stack], start=2, stop=5)[0]
        assert "Stack Slicer" in out.process_history_string()


    def test_outputs_of_one_call_share_all_id(self):
        first, second = make_stack(seed=44), make_stack(seed=45)

        out = stack_slice_acquisitions([first, second], start=0, stop=3)

        assert out[0].get_history()["call_id"] == out[1].get_history()["call_id"]
        assert [stack.get_history()["output_index"] for stack in out] == [0, 1]
        assert all(stack.get_history()["output_count"] == 2 for stack in out)


class TestFrameTimes:

    def test_shape_retention(self, synthetic_stack):
        times = synthetic_stack.times_of_flight()
        assert times.shape == (6,)
        assert times.dtype == np.float64

    def test_timeless_stack(self):
        assert make_stack(times=False).times_of_flight() is None

    def test_time_setter(self):
        stack = make_stack(times=False)
        stack.set_times_of_flight(np.linspace(0.0, 0.1, 6))
        assert stack.times_of_flight()[-1] == pytest.approx(0.1)

    def test_time_setter_overwrite(self, synthetic_stack):
        synthetic_stack.set_times_of_flight(np.zeros(6))
        assert np.all(synthetic_stack.times_of_flight() == 0.0)

    @requires_sample_data
    def test_spectra_derivation(self, sample_stack):
        times = sample_stack.times_of_flight()
        assert times.shape == (sample_stack.data.shape[0],)
        assert times[0] == pytest.approx(0.00474)

    def test_collimation_distance(self, synthetic_stack):
        wavelengths = synthetic_stack.wavelengths(collimation_distance=50.0)
        expected = frame_wavelengths(
            synthetic_stack.times_of_flight(), 0.0, 50.0, apply_delay=True
        )
        assert np.allclose(wavelengths, expected)

    def test_delay(self, synthetic_stack):
        undelayed = synthetic_stack.wavelengths(0.0, 50.0)
        delayed = synthetic_stack.wavelengths(0.01, 50.0, apply_delay=True)
        ignored = synthetic_stack.wavelengths(0.01, 50.0, apply_delay=False)

        assert np.all(delayed > undelayed)
        assert np.allclose(ignored, undelayed)

    def test_derived_wavelengths_are_recorded_on_stack(self, synthetic_stack):
        derived = synthetic_stack.wavelengths(collimation_distance=50.0)
        assert np.allclose(synthetic_stack.stack_meta["wavelengths"], derived)

    def test_no_times_no_wavelengths(self):
        assert make_stack(times=False).wavelengths(collimation_distance=50.0) is None


class TestAnalysisResults:
    """Non-array results attached by the analysis processes."""

    def test_results_roundtrip(self):
        stack = make_stack()
        stack.record_analysis_results({"mean": 1.5, "count": 3})
        assert stack.analysis_results() == {"mean": 1.5, "count": 3}

    def test_append_results(self):
        stack = make_stack()
        stack.record_analysis_results({"mean": 1.5})
        stack.record_analysis_results({"median": 2.0})
        assert stack.analysis_results() == {"mean": 1.5, "median": 2.0}

    def test_key_update(self):
        stack = make_stack()
        stack.record_analysis_results({"mean": 1.5})
        stack.record_analysis_results({"mean": 9.0})
        assert stack.analysis_results() == {"mean": 9.0}

    def test_empty_results_change_nothing(self):
        stack = make_stack()
        stack.record_analysis_results({})
        assert stack.analysis_results() == {}

    def test_results_getter(self):
        stack = make_stack()
        stack.record_analysis_results({"mean": 1.5})
        stack.analysis_results()["mean"] = 0.0
        assert stack.analysis_results() == {"mean": 1.5}

    def test_results_string(self):
        stack = make_stack()
        stack.record_analysis_results({"mean": 1.5, "label": "roi"})
        text = stack.analysis_results_text()
        assert "mean: 1.5" in text
        assert "label: roi" in text

    def test_absent_results_string(self):
        assert make_stack().analysis_results_text() == "(no analysis results)"
