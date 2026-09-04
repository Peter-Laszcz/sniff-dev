"""
Functions for reading and writing full-processing workflows to/from files.
Saved workflows hold process tree and parameters but no stacks.
These must be provided by the importing user upon replay.

File layout (JSON/UTF-8):

    {
      "format":         "SNIFF_WORKFLOW",
      "name":           (run name),
      "nodes": list[ # See components/workflow/WorkflowNode for more details
        {
          "uuid": str,
          "name": str, # stack display name
          "process": str | null, # null for an entry point
          "params": {...}, # process parameter dictionary
          "inputs": [str, ...], # parent uuids
          "aux": {role: UUID (str)}, # auxiliary uuids
          "mode": "map" | "reduce",
          "call_id" :str | null, # shared by one process' outputs
          "output_index": int,
          "output_count": int
        },
        ... ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from sni_app.core.components.workflow import WorkflowGraph, WorkflowNode
from sni_app.core.io.project import _decode_meta_value, _encode_meta_value

_WORKFLOW_FORMAT = "SNIFF_WORKFLOW"
"""
Checks we are working with the right format of JSON
"""


def _encode_params(params: Optional[dict], process: str) -> dict:
    """
    Encode one node's parameters, raising if encoding fails.
    """
    encoded = {}
    for key, value in (params or {}).items():
        try:
            encoded[str(key)] = _encode_meta_value(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Cannot write parameter {key!r} of process {process!r} to a "
                f"workflow file: {exc}"
            ) from exc
    return encoded


def _workflow_to_dict(graph: WorkflowGraph, name: str = "") -> dict:
    """
    Render graph as a serialisable dictionary for writing.

    Parameters
    ----------
    graph : WorkflowGraph
        The workflow to export.
    name : str, optional
        A human-readable name for the workflow; defaults to the graph's own.

    Returns
    -------
    dict
        The workflow in serialisable format
    """
    nodes = []
    for nid in graph.order():  # parents before children, for readability
        node = graph.nodes[nid]
        nodes.append(
            {
                "uuid": node.uuid,
                "name": node.name,
                "process": node.process,
                "params": _encode_params(node.params, node.process or "(entry point)"),
                "inputs": list(node.inputs),
                "aux": dict(node.aux),
                "mode": node.mode,
                "call_id": node.call_id,
                "output_index": int(node.output_index),
                "output_count": int(node.output_count),
            }
        )
    return {
        "format": _WORKFLOW_FORMAT,
        "name": str(name or graph.name or ""),
        "nodes": nodes,
    }


def _workflow_from_dict(payload: dict) -> WorkflowGraph:
    """
    Rebuild a WorkflowGraph from a workflow dictionary.

    Parameters
    ----------
    payload : dict
        A dict of the format that _workflow_to_dict produces.

    Returns
    -------
    WorkflowGraph

    Raises
    ------
    ValueError
        If the dictionary is invalid.
    """
    if not isinstance(payload, dict):
        raise ValueError("Not a SNIFF workflow file (expected a JSON object).")

    fmt = payload.get("format")
    if fmt != _WORKFLOW_FORMAT:
        raise ValueError(f"Not a SNIFF workflow file (format={fmt!r}).")

    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Workflow file has no 'nodes' list.")

    nodes: dict[str, WorkflowNode] = {}
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise ValueError("Workflow file contains a malformed node entry.")
        nid = raw.get("id")
        if not isinstance(nid, str) or not nid:
            raise ValueError("Workflow file contains a node with no id.")
        params = {
            str(k): _decode_meta_value(v) for k, v in (raw.get("params") or {}).items()
        }
        inputs = [str(v) for v in (raw.get("inputs") or [])]
        aux = {str(k): str(v) for k, v in (raw.get("aux") or {}).items()}
        process = raw.get("process")
        nodes[nid] = WorkflowNode(
            uuid=nid,
            stack=None,  # a workflow file carries no stacks
            name=str(raw.get("name") or "stack"),
            process=str(process) if process else None,
            params=params,
            inputs=inputs,
            aux=aux,
            mode=str(raw.get("mode") or ""),
            call_id=raw.get("call_id"),
            output_index=int(raw.get("output_index", 0)),
            output_count=int(raw.get("output_count", 1)),
            is_entry=not inputs and not aux,
            is_ghost=False,
        )

    for node in nodes.values():
        for pid in node.parents:
            if pid not in nodes:
                raise ValueError(
                    f"Workflow file is incomplete: step '{node.name}' needs an "
                    f"input ({pid}) that the file does not describe."
                )

    return WorkflowGraph(nodes=nodes, name=str(payload.get("name") or ""))


def save_workflow(path, graph: WorkflowGraph, name: str = "") -> Path:
    """
    Write graph to path as a workflow file.

    Parameters
    ----------
    path : str or Path
        Destination file. Workflow suffix is appended when the path has
        no suffix.
    graph : WorkflowGraph
        The workflow to export.
    name : str, optional
        Human-readable name of workflow.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If graph has no nodes.
    TypeError
        If a process parameter cannot be encoded.
    """
    if graph is None or not graph.nodes:
        raise ValueError("Nothing to export: the workflow has no steps.")
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")
    payload = _workflow_to_dict(graph, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_workflow(path) -> WorkflowGraph:
    """
    Read a workflow file written by save_workflow.

    Parameters
    ----------
    path : str or Path
        The workflow file to read.

    Returns
    -------
    WorkflowGraph
        The workflow.

    Raises
    ------
    ValueError
        If the file is not readable as a SNIFF workflow.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc
    return _workflow_from_dict(payload)


def save_workflows(path, graphs: Sequence[WorkflowGraph], name: str = "") -> list:
    """
    Write each given graph to its own file alongside within the given path.

    Parameters
    ----------
    path : str or Path
        Destination for the works.
    graphs : sequence of WorkflowGraph
        The workflows to export.
    name : str, optional
        Human-readable name.

    Returns
    -------
    list of Path
        The paths written, in order.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")
    written = []
    for i, graph in enumerate(graphs, 1):
        target = path if i == 1 else path.with_name(f"{path.stem}_{i}{path.suffix}")
        written.append(save_workflow(target, graph, name))
    return written
