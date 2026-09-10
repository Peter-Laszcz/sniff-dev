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
Workflow graphs built from stack provenance, and their replay.
"""

import numpy as np
import pytest

from sni_app.core import (
    PROCESS_REGISTRY,
    Stack,
    WorkflowGraph,
    active_stacks,
    aux_roles_of,
    default_entry_map,
    entry_point_uuids,
    get_process_spec,
    load_workflow,
    replay_workflow,
    sanitise_params,
    save_workflow,
    save_workflows,
    stack_bin_frames,
    stack_slice_acquisitions,
    stack_sum,
    workflow_mode,
)

from conftest import make_stack


@pytest.fixture
def chain() -> tuple:
    """A two-step pipeline: slice, then sum. Returns (entry, sliced, summed)."""
    entry = make_stack(n_frames=6, seed=300)
    entry.robust_stack_uuid()
    sliced = stack_slice_acquisitions([entry], start=0, stop=4)[0]
    summed = stack_sum([sliced])[0]
    return entry, sliced, summed


class TestGraphBuilding:
    """Grouping stacks into workflows."""

    def test_pipeline_forms_one_graph(self, chain):
        graphs = WorkflowGraph.from_stacks(list(chain))

        assert len(graphs) == 1
        assert graphs[0].size() == 3

    def test_disjoint_pipelines_form_separate_graphs(self, chain):
        other_entry = make_stack(seed=301)
        other_entry.robust_stack_uuid()
        other = stack_slice_acquisitions([other_entry], start=0, stop=2)[0]

        graphs = WorkflowGraph.from_stacks([*chain, other_entry, other])

        assert len(graphs) == 2
        assert [graph.size() for graph in graphs] == [3, 2]

    def test_graph_list_ordering(self, chain):
        other_entry = make_stack(seed=302)
        other_entry.robust_stack_uuid()
        other = stack_slice_acquisitions([other_entry], start=0, stop=2)[0]

        graphs = WorkflowGraph.from_stacks([other_entry, other, *chain])

        assert graphs[0].size() > graphs[1].size()

    def test_ghost_node(self, chain):
        _entry, sliced, summed = chain

        graph = WorkflowGraph.from_stacks([sliced, summed])[0]

        ghosts = [node for node in graph.nodes.values() if node.is_ghost]
        assert [node.name for node in ghosts] == ["(removed stack)"]

    def test_lone_stack_exclusion(self):
        """A stack that is nobody's parent or child is in no workflow."""
        assert active_stacks([make_stack(seed=303)]) == []

    def test_a_parent_of_a_processed_stack_is_active(self, chain):
        entry, sliced, _summed = chain
        assert entry in active_stacks([entry, sliced])


class TestGraphNavigation:
    """Reading a graph's structure."""

    def test_entry_point_is_unprocessed(self, chain):
        entry, _sliced, _summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        assert entry_point_uuids(graph) == [entry.stack_uuid()]

    def test_entry_map_stack_getter(self, chain):
        entry, _sliced, _summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        assert default_entry_map(graph) == {entry.stack_uuid(): entry}

    def test_parent_child_derivation(self, chain):
        entry, sliced, _summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        assert graph.node_children(entry.stack_uuid()) == [sliced.stack_uuid()]

    def test_depth_functionality(self, chain):
        entry, sliced, summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        assert graph.node_depth(entry.stack_uuid()) == 0
        assert graph.node_depth(sliced.stack_uuid()) == 1
        assert graph.node_depth(summed.stack_uuid()) == 2

    def test_graph_node_order(self, chain):
        entry, sliced, summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        order = graph.order()
        assert order.index(entry.stack_uuid()) < order.index(sliced.stack_uuid())
        assert order.index(sliced.stack_uuid()) < order.index(summed.stack_uuid())

    def test_label_string(self, chain):
        graph = WorkflowGraph.from_stacks(list(chain))[0]
        assert graph.label() == "stack  (3 stacks)"


class TestProcessRegistry:
    """The processes a workflow can replay. Heavy on tests as
    this is the expected entry points for new processes and plugins.
    """

    def test_process_modes(self):
        assert all(
            spec.mode in ("map", "reduce") for spec in PROCESS_REGISTRY.values()
        )

    def test_process_names(self):
        assert get_process_spec("Stack Slicer").mode == "map"

    def test_unknown_process_has_no_spec(self):
        assert get_process_spec("Not A Process") is None

    def test_unknown_process_default_mode(self):
        assert workflow_mode("Not A Process") == "map"

    def test_auxiliary_roles_reported(self):
        assert aux_roles_of("Normalisation") == ("open_beam",)
        assert aux_roles_of("Stack Slicer") == ()

    @pytest.mark.parametrize(
        "process",
        ["Stack Slicer", "Stack Summation", "Normalisation", "Overlap Correction"],
    )
    def test_process_replayability(self, process):
        """Every name written into history must be replayable."""
        assert process in PROCESS_REGISTRY


class TestSanitiseParams:
    """Making recorded parameters serialisable."""

    def test_stacks_are_dropped(self):
        assert sanitise_params({"open_beam": make_stack(seed=304), "scale": 2.0}) == {
            "scale": 2.0
        }

    def test_keys_become_strings(self):
        assert sanitise_params({1: "a"}) == {"1": "a"}

    def test_nothing_in_nothing_out(self):
        assert sanitise_params(None) == {}
        assert sanitise_params({}) == {}


class TestReplay:
    """Running a recorded workflow again on fresh input."""

    def test_replay_runs(self, chain):
        entry, _sliced, summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]
        fresh = make_stack(n_frames=6, seed=305)

        produced, errors = replay_workflow(graph, {entry.stack_uuid(): fresh})

        assert errors == {}
        replayed = produced[summed.stack_uuid()]
        assert replayed.data.shape == summed.data.shape
        assert np.allclose(replayed.data, fresh.data[:4], rtol=1e-5)

    def test_replay_is_correct(self, chain):
        entry, _sliced, summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        produced, errors = replay_workflow(graph, {})

        assert errors == {}
        assert np.allclose(produced[summed.stack_uuid()].data, summed.data)

    def test_stackless_entry_point_reported(self, chain):
        _entry, sliced, summed = chain
        graph = WorkflowGraph.from_stacks([sliced, summed])[0]
        ghost = next(node for node in graph.nodes.values() if node.is_ghost)

        _produced, errors = replay_workflow(graph, {})

        assert "no stack supplied" in errors[ghost.uuid]

    def test_failing_step_reported(self):
        """A parameter the process rejects is recorded as that node's error."""
        entry = make_stack(n_frames=6, seed=306)
        entry.robust_stack_uuid()
        binned = stack_bin_frames([entry], bin_factor=2, he_le=None)[0]
        graph = WorkflowGraph.from_stacks([entry, binned])[0]
        graph.nodes[binned.stack_uuid()].params["bin_factor"] = 99

        _produced, errors = replay_workflow(graph, {entry.stack_uuid(): entry})

        assert "No complete bins" in errors[binned.stack_uuid()]


class TestWorkflowFiles:
    """Saving and loading workflows."""

    def test_node_roundtrip(self, chain, tmp_path):
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        path = save_workflow(tmp_path / "flow", graph, name="two step")
        loaded = load_workflow(path)

        assert path.suffix == ".json"
        assert loaded.size() == graph.size()
        assert loaded.name == "two step"

    def test_parameter_round_trip(self, chain, tmp_path):
        _entry, sliced, _summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        loaded = load_workflow(save_workflow(tmp_path / "flow.json", graph))

        assert loaded.nodes[sliced.stack_uuid()].params == {"start": 0, "stop": 4}
        assert loaded.nodes[sliced.stack_uuid()].process == "Stack Slicer"

    def test_file_is_lightweight(self, chain, tmp_path):
        graph = WorkflowGraph.from_stacks(list(chain))[0]

        loaded = load_workflow(save_workflow(tmp_path / "flow.json", graph))

        assert all(node.stack is None for node in loaded.nodes.values())

    def test_empty_workflow_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            save_workflow(tmp_path / "empty.json", WorkflowGraph())

    def test_wrong_format_raises_error(self, tmp_path):
        path = tmp_path / "junk.json"
        path.write_text("not json at all", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_workflow(path)

    def test_workflow_write(self, chain, tmp_path):
        other_entry = make_stack(seed=307)
        other_entry.robust_stack_uuid()
        other = stack_slice_acquisitions([other_entry], start=0, stop=2)[0]
        graphs = WorkflowGraph.from_stacks([*chain, other_entry, other])

        written = save_workflows(tmp_path / "flows", graphs)

        assert len(written) == 2
        assert all(path.exists() for path in written)
        assert len({path.name for path in written}) == 2

    def test_loaded_workflow_replay(self, chain, tmp_path):
        entry, _sliced, summed = chain
        graph = WorkflowGraph.from_stacks(list(chain))[0]
        loaded = load_workflow(save_workflow(tmp_path / "flow.json", graph))
        fresh = make_stack(n_frames=6, seed=308)

        produced, errors = replay_workflow(loaded, {entry.stack_uuid(): fresh})

        assert errors == {}
        assert isinstance(produced[summed.stack_uuid()], Stack)
