"""
Contains workflow data  structure, creation methods, and process replay functionality.
Stacks processed through SNIFF have their unique provenance tracked throughout processing,
enabling the construction of a representative digraph of the dataset's processing history.
This can be divided into disjoint workflows (digraphs whose stacks do not interact in a meaningful way).
In conjunction with recorded process parameters, these workflows are able to be replayed on supplied "entry stacks".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from sni_app.core.components.stack import Stack, record_derivation
from sni_app.core.process.roi_processes import (
    atten_coefficient,
    h_cross_section,
    relative_attenuation,
    roi_to_stack,
    sum_of_logs_relative_attenuation,
    t_cross_section,
)
from sni_app.core.process.stack_processes import (
    OVERLAP_ROLES,
    stack_avg,
    stack_bin_frames,
    stack_join,
    stack_normalisation,
    stack_overlap_correction,
    stack_registration,
    stack_sbkg_correction,
    stack_scrubbing,
    stack_slice_acquisitions,
    stack_stitching,
    stack_sum,
)

_log = logging.getLogger("SNIFF_Log")


#############
# THE GRAPH #
#############


@dataclass
class WorkflowNode:
    """
    One stack in the workflow graph.

    Attributes
    ----------
    uuid : str
        The stack's unique identifier.
    stack : object or None
        The node's stack, or None if the node is a ghost node
        (i.e. original stack is irretrievable; used in workflow import)
    name : str
        Display name.
    process : str or None
        UUID of the process that produced this stack; None for an entry point.
    params : dict
        Process parameter dictionary (empty for entry points / ghosts).
    inputs : list[str]
        UUIDs of the primary parent stacks.
    aux : dict{str:str}
        {aux stack role :  uuid of aux}.
    mode : str
        Function type of process. "map" (one-to-one function) or "reduce" (many-to-one function).
    call_id : str
        Identifier shared by the outputs of a single process.
    output_index : int
        Output index of stack in previous process output.
    output_count : int
        Number of output stacks in previous process output.

    is_entry : bool
        Whether this is a workflow entry point (a root with a real stack).
    is_ghost : bool
        Whether this node has no backing stack.
    """

    uuid: str
    stack: object = None
    name: str = ""
    process: Optional[str] = None
    params: dict = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    aux: dict[str, str] = field(default_factory=dict)
    mode: str = ""
    call_id: Optional[str] = None
    output_index: int = 0
    output_count: int = 1
    is_entry: bool = False
    is_ghost: bool = False

    @property
    def parents(self) -> list[str]:
        """All parent ids (primary inputs followed by auxiliary inputs)."""
        return list(self.inputs) + list(self.aux.values())

    @classmethod
    def from_stack(cls, stack: Stack) -> WorkflowNode:
        """Build a WorkflowNode from a live stack and its history."""
        stack_uuid = stack.robust_stack_uuid()
        history = stack.get_history()
        if not history:
            return WorkflowNode(
                uuid=stack_uuid, stack=stack, name=stack.display_name(), is_entry=True
            )
        return cls(
            uuid=stack_uuid,
            stack=stack,
            name=stack.display_name(),
            process=history.get("process"),
            params=dict(history.get("params", {})),
            inputs=list(history.get("inputs", [])),
            aux=dict(history.get("aux", {})),
            mode=history.get("mode", ""),
            call_id=history.get("call_id"),
            output_index=int(history.get("output_index", 0)),
            output_count=int(history.get("output_count", 1)),
            is_entry=not history.get("inputs") and not history.get("aux"),
        )


@dataclass
class WorkflowGraph:
    """
    One connected workflow.

    Attributes
    ----------
    nodes : dict[str,WorkflowNode]
        Every node in the graph, keyed by UUID.
    name : str
        Optional human-readable name; used upon workflow import.
    """

    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    name: str = ""

    @classmethod
    def from_stacks(cls, stacks: list[Stack]) -> list[WorkflowGraph]:
        """
        Build all workflow graphs from a list of stacks.

        Parameters
        ----------
        stacks : list[Stack]
            The stacks to graph.

        Returns
        -------
        list of WorkflowGraph
            One graph per disjoint workflow, ordered by largest first.
            Absent stacks are referenced by ghost nodes to avoid accidental discontinuity.
        """
        nodes: dict[str, WorkflowNode] = {}

        # build dict of nodes per stack (also enables multi-node membership)
        for stack in active_stacks(stacks):
            node = WorkflowNode.from_stack(stack)
            nodes[node.uuid] = node

        # ghost nodes
        for node in list(nodes.values()):
            for parent_uuid in node.parents:
                if parent_uuid not in nodes:
                    nodes[parent_uuid] = WorkflowNode(
                        uuid=parent_uuid,
                        stack=None,
                        name="(removed stack)",
                        is_ghost=True,
                    )

        # build adjacency map for each node
        adjacency: dict[str, set] = {nid: set() for nid in nodes}
        for node in nodes.values():
            for parent_uuid in node.parents:
                if parent_uuid in nodes:
                    adjacency[node.uuid].add(parent_uuid)
                    adjacency[parent_uuid].add(node.uuid)

        # assign nodes to workflows
        seen: set = set()
        components: list[WorkflowGraph] = []
        for start in nodes:
            if start in seen:  # already processed
                continue
            stack_ids = [start]
            member: set = set()
            while stack_ids:
                node_uuid = stack_ids.pop()
                if node_uuid in member:
                    continue
                member.add(node_uuid)
                seen.add(node_uuid)
                stack_ids.extend(adjacency[node_uuid] - member)
            components.append(cls(nodes={nid: nodes[nid] for nid in member}))

        # sort by graph size
        components.sort(key=lambda g: (-g.size(), g.label()))
        return components

    def entry_points(self) -> list[WorkflowNode]:
        """Entry points in order."""
        rs = [n for n in self.nodes.values() if not n.parents]
        return sorted(rs, key=lambda n: (n.name.lower(), n.uuid))

    def node_children(self, node_uuid: str) -> list[str]:
        """UUIDs of nodes listing node_uuid among their parents."""
        return [n.uuid for n in self.nodes.values() if node_uuid in n.parents]

    def node_depth(self, node_id: str, _seen: Optional[set] = None) -> int:
        """
        Longest path length from any entry point to node given by node_id (entry points are depth 0). Recursive.
        """
        node = self.nodes.get(node_id)
        if node is None or not node.parents:
            return 0
        _seen = _seen or set()
        if node_id in _seen:
            return 0
        _seen = _seen | {node_id}  # exclude node from recursive call
        present = [p for p in node.parents if p in self.nodes]
        if not present:
            return 0
        return 1 + max(
            self.node_depth(p, _seen) for p in present
        )  # increments 1 per level till all instances return 0

    def order(self) -> list[str]:
        """
        Return node ids in dependency order (parent then child). Leftover nodes appended at the end
        (facilitates entry point assignment)
        """
        indegrees = {
            node_uuid: sum(1 for parent in node.parents if parent in self.nodes)
            for node_uuid, node in self.nodes.items()
        } # per-node indegrees
        queue = [nid for nid, d in indegrees.items() if d == 0]
        order: list[str] = []
        while queue:
            queue.sort()
            node_uuid = queue.pop(0)
            order.append(node_uuid)
            for child in self.node_children(node_uuid):
                indegrees[child] -= 1
                if indegrees[child] == 0:
                    queue.append(child)
        # Append stragglers
        for node_uuid in self.nodes:
            if node_uuid not in order:
                order.append(node_uuid)
        return order

    def size(self) -> int:
        """Number of nodes in graph."""
        return len(self.nodes)

    def label(self) -> str:
        """Readable label for graph"""
        if self.name:
            return f"{self.name}  ({self.size()} stacks)"
        roots = self.entry_points()
        root_name = roots[0].name if roots else "workflow"
        extra = f" +{len(roots) - 1}" if len(roots) > 1 else ""
        return f"{root_name}{extra}  ({self.size()} stacks)"


def active_stacks(stacks: list[Stack]) -> list[Stack]:
    """
    Return the subset of stacks that are active in some workflow.
    A stack is considered active if it is a parent or child of another stack.

    Parameters
    ----------
    stacks : sequence of Stack
        Stacks to filter.

    Returns
    -------
    list of Stack
        Active stacks.
    """
    referenced: set = set()
    for stack in stacks:
        referenced.update(stack.parent_ids())
    return [
        stack
        for stack in stacks
        if not stack.is_entry_point() or stack.stack_uuid() in referenced
    ]



RunFn = Callable[[list[Stack], dict[str, Stack], dict], list[Stack]]
"""Inputs: (input stacks, auxiliary stack dict, parameter dict), Outputs: (output stacks)"""

@dataclass(frozen=True)
class ProcessSpec:
    """One replayable process. RunFn is defined and explained above."""

    mode: str
    run: RunFn
    aux_roles: tuple = field(default_factory=tuple)


####################
#   RUN ADAPTORS   #
####################


def _run_overlap(inputs, aux, params) -> list[Stack]:
    # "(internal)" means the stack's stored metadata was used, i.e. no external parameters were supplied.
    overrides = {
        role: ("" if params.get(role) in (None, "(internal)") else params[role])
        for role in OVERLAP_ROLES
    }
    return stack_overlap_correction(inputs, **overrides)


def _run_normalisation(inputs, aux, params) -> list[Stack]:
    return stack_normalisation(inputs, aux["open_beam"], **params)


def _run_scrubbing(inputs, aux, params) -> list[Stack]:
    params = dict(params)
    open_beam_dir = params.pop("open_beam_dir", "") or ""
    weights = inputs[0].stack_meta.get("weights_data_frame") if inputs else None
    if weights is None and not open_beam_dir:
        raise ValueError(
            "Scrubbing correction needs either a weights dataframe on the input "
            "stack or a recorded open-beam folder."
        )
    return stack_scrubbing(inputs, weights, open_beam_dir=open_beam_dir, **params)


def _run_slicer(inputs, aux, params) -> list[Stack]:
    return stack_slice_acquisitions(inputs, **params)


def _run_avg(inputs, aux, params) -> list[Stack]:
    return stack_avg(inputs)


def _run_sum(inputs, aux, params) -> list[Stack]:
    return stack_sum(inputs)


def _run_bin(inputs, aux, params) -> list[Stack]:
    return stack_bin_frames(inputs, **params)


def _run_join(inputs, aux, params) -> list[Stack]:
    return stack_join(inputs, **params)


def _run_registration(inputs, aux, params) -> list[Stack]:
    return stack_registration(inputs, aux["reference"], **params)


def _run_sbkg(inputs, aux, params) -> list[Stack]:
    return stack_sbkg_correction(inputs, aux["bb_mask"], **params)


def _run_stitching(inputs, aux, params) -> list[Stack]:
    return stack_stitching(inputs[0], inputs[1], **params)


def _run_roi_to_stack(inputs, aux, params) -> list[Stack]:
    return roi_to_stack(inputs, params.get("roi_xywh", [0, 0, 0, 0]))


def _run_relatt_standard(inputs, aux, params) -> list[Stack]:
    return relative_attenuation(inputs[0], **params)


def _run_relatt_sol(inputs, aux, params) -> list[Stack]:
    return sum_of_logs_relative_attenuation(inputs[0], **params)


def _run_atten_coefficient(inputs, aux, params) -> list[Stack]:
    return atten_coefficient(inputs[0], inputs[1], **params)


def _run_t_cross_section(inputs, aux, params) -> list[Stack]:
    return t_cross_section(inputs[0], **params)


def _run_h_cross_section(inputs, aux, params) -> list[Stack]:
    return h_cross_section(inputs[0], inputs[1], **params)


####################
# PROCESS REGISTRY #
####################

PROCESS_REGISTRY: dict[str, ProcessSpec] = {
    "Overlap Correction": ProcessSpec("map", _run_overlap),
    "Normalisation": ProcessSpec("map", _run_normalisation, ("open_beam",)),
    "Scrubbing Correction": ProcessSpec("map", _run_scrubbing),
    "SBKG Correction": ProcessSpec("map", _run_sbkg, ("bb_mask",)),
    "Stack Slicer": ProcessSpec("map", _run_slicer),
    "Stack Averaging": ProcessSpec("reduce", _run_avg),
    "Stack Summation": ProcessSpec("reduce", _run_sum),
    "Bin Stack Frames": ProcessSpec("map", _run_bin),
    "Join Stacks": ProcessSpec("reduce", _run_join),
    "Stack Registration": ProcessSpec("map", _run_registration, ("reference",)),
    "Stack Stitching": ProcessSpec("reduce", _run_stitching),
    "ROI to Stack": ProcessSpec("map", _run_roi_to_stack),
    "Relative Attenuation (images)": ProcessSpec("reduce", _run_relatt_standard),
    "Rel. Attenuation (sum-of-logs)": ProcessSpec("reduce", _run_relatt_sol),
    "Attenuation Coefficient": ProcessSpec("reduce", _run_atten_coefficient),
    "Total Microscopic Cross Section": ProcessSpec("reduce", _run_t_cross_section),
    "Hydrogen Cross Section": ProcessSpec("reduce", _run_h_cross_section),
}
"""
Every replayable process, keyed by the name it is stored as within process history.
"""


def get_process_spec(process: str):
    """Return the ProcessSpec for a given process"""
    return PROCESS_REGISTRY.get(process)


def workflow_mode(process: str) -> str:
    """Return the recorded output mode for a given process"""
    spec = PROCESS_REGISTRY.get(process)
    return spec.mode if spec is not None else "map"  # default


def aux_roles_of(process: str) -> tuple[str, ...]:
    """Return the auxiliary-input role names for a process."""
    spec = PROCESS_REGISTRY.get(process)
    return spec.aux_roles if spec is not None else tuple()


##########
# REPLAY #
##########


def entry_point_uuids(graph: WorkflowGraph) -> list[str]:
    """UUIDs of the graph's entry points (roots the user supplies stacks for)."""
    return [n.uuid for n in graph.entry_points()]


def default_entry_map(graph: WorkflowGraph) -> dict[str, object]:
    """
    Map each entry-point id to its current backing stack, if one exists.
    Provides default for workflows that happen to have original stacks.
    """
    return {
        n.uuid: n.stack
        for n in graph.entry_points()
        if getattr(n, "stack", None) is not None
    }


def _invocations(graph: WorkflowGraph) -> dict[str, list[WorkflowNode]]:
    """Group the graph's produced nodes by call_id."""
    calls: dict[str, list[WorkflowNode]] = {}
    for node in graph.nodes.values():
        if node.is_entry or node.is_ghost or not node.call_id:
            continue
        calls.setdefault(node.call_id, []).append(node)
    for group in calls.values():
        group.sort(key=lambda n: n.output_index)
    return calls


def replay_workflow(
    graph: WorkflowGraph, entry_stacks: dict[str, Stack]
) -> tuple[dict[str, Stack], dict[str, str]] | None:
    """
    Replay graph using given entry-point stacks.

    Parameters
    ----------
    graph : WorkflowGraph
        The workflow component to replay.
    entry_stacks : dict[str,Stack]
        Stack to supply for each entry-point.
        Absent entry points use their own stacks if stored, fails otherwise.

    Returns
    -------
    tuple[dict[str, Stack], dict[str, str]] | None
        ({output uuid: processed Stack}, {output uuid: error message}).
    """
    errors = {}
    produced = {}

    # fill entry points
    for node in graph.nodes.values():
        if node.parents:
            continue  # not an entry point
        supplied = entry_stacks.get(node.uuid, None)
        if supplied is None:
            supplied = getattr(
                node, "stack", None
            )  # attempt population with internal stack
        if supplied is None:
            errors[node.uuid] = "no stack supplied for this entry point"
            continue
        produced[node.uuid] = supplied

    invocations = _invocations(graph)
    done: set = set()  # avoid rerunning nodes

    for node_uuid in graph.order():
        node = graph.nodes.get(node_uuid)
        if node is None or node_uuid in produced or node_uuid in errors:
            continue
        if node.is_ghost or not node.call_id or node.call_id in done:
            continue

        group = invocations.get(node.call_id, [node])
        input_ids = _group_input_ids(node, group)
        needed = input_ids + list(
            node.aux.values()
        )  # ordered inputs of node followed by aux stacks

        if any(parent_uuid in errors for parent_uuid in needed):
            done.add(node.call_id)
            for member in group:
                errors[member.uuid] = "(an earlier step failed)"
            continue
        if any(pid not in produced for pid in needed):
            continue

        done.add(node.call_id)
        produced, errors = _replay_process(node, group, input_ids, produced, errors)
    return produced, errors


def _group_input_ids(node: WorkflowNode, group: list[WorkflowNode]) -> list[str]:
    """The ordered primary input ids of an invocation."""
    if node.mode == "reduce":
        return list(node.inputs)
    # map: one input per output, ordered by output_index
    return [m.inputs[0] for m in group if m.inputs]


def _replay_process(
    node: WorkflowNode,
    group: list[WorkflowNode],
    input_ids: list[str],
    produced: dict[str, Stack],
    errors: dict[str, str],
) -> tuple[dict[str, Stack], dict[str, str]] | None:
    """Reproduce one process invocation (all outputs sharing a call_id)."""
    process = node.process
    spec = PROCESS_REGISTRY.get(process)
    if spec is None:
        for member in group:
            errors[member.uuid] = f"no replay handler for process '{process}'"
        return produced, errors

    input_stacks = [produced[pid] for pid in input_ids]
    aux_stacks = {role: produced[pid] for role, pid in node.aux.items()}

    try:
        outputs = spec.run(input_stacks, aux_stacks, dict(node.params))
    except Exception as exc:  # a computation failing must not abort the whole replay
        for member in group:
            errors[member.uuid] = f"replay failed: {exc}"
        return produced, errors

    if any(out.get_history() is None for out in outputs):
        record_derivation(
            outputs, process, node.params, input_stacks, aux=aux_stacks, mode=node.mode
        )

    for member in group:
        idx = member.output_index
        if idx < len(outputs):
            output = outputs[idx]
            output.stack_meta["display_name"] = member.name
            produced[member.uuid] = output
        else:
            errors[member.uuid] = "process produced fewer outputs than recorded"
    return produced, errors


def sanitise_params(params: Optional[Mapping]) -> dict:
    """
    Return a shallow copy of stack parameters, removing stacks (enables serialisation).
    """
    if not params:
        return {}
    return {str(k): v for k, v in params.items() if not isinstance(v, Stack)}
