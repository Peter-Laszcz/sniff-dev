"""
Project saving/loading functionality. (De)serialises to/from .sniff (HDF5) files.
A project is a standalone file storing loaded stacks, GUI states, and GUI process parameters.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

from sni_app.core.components.stack import Stack


@dataclass
class Project:
    """
    Class returned from loading project file.

    Attributes
    ----------
    stacks : List[Stack]
        Stacks in stack list. User defined GUI values held in stack_meta under
        'display_name' and 'selected_for_processing'.
    gui_state : dict
        GUI field values (source directory, selected function, etc).
    _file : Optional[h5py.File]
        HDF5 handler, if not yet released.
    """

    stacks: List[Stack] = field(default_factory=list)
    gui_state: dict = field(default_factory=dict)
    _file: Optional[h5py.File] = None

    def close(self) -> None:
        """
        Release HDF5 handler.
        """
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None


#####################
# ENCODING/DECODING #
#####################


def _encode_pandas_index(index) -> dict:
    """
    Encode a pandas Index as encoded values plus its name.
    """
    name = index.name
    return {
        "values": [_encode_meta_value(v) for v in index.tolist()],
        "name": name if (name is None or isinstance(name, str)) else str(name),
    }


def _decode_pandas_index(spec) -> pd.Index:
    values = [_decode_meta_value(v) for v in (spec or {}).get("values", [])]
    return pd.Index(values, name=(spec or {}).get("name"))


def _encode_meta_value(value):
    """
    Convert value into a serialisable type. Operates recursively on value storing values.
    Already serialisable values are not changed. Changed values are placed into a dictionary alongside
    their original type so that they can be changed back upon deserialisation.

    Serialisation methods:
        Numpy arrays: nested list and dtype
        DataFrames: per column values and dtype
        Path: str(Path)
        set/tuple : list

    Raises
    ------
    TypeError
        If value cannot be encoded.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "__sniff_type__": "ndarray",
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "__sniff_type__": "dataframe",
            "index": _encode_pandas_index(value.index),
            "columns": _encode_pandas_index(value.columns),
            "frame": [
                {
                    "dtype": str(value.dtypes.iloc[i]),
                    "values": [
                        _encode_meta_value(v) for v in value.iloc[:, i].tolist()
                    ],
                }
                for i in range(value.shape[1])
            ],
        }
    if isinstance(value, pd.Series):
        return {
            "__sniff_type__": "series",
            "index": _encode_pandas_index(value.index),
            "name": (
                value.name
                if (value.name is None or isinstance(value.name, str))
                else str(value.name)
            ),
            "dtype": str(value.dtype),
            "values": [_encode_meta_value(v) for v in value.tolist()],
        }
    if isinstance(value, Path):
        return {"__sniff_type__": "path", "data": str(value)}
    if isinstance(value, (set, frozenset)):
        return {"__sniff_type__": "set", "data": [_encode_meta_value(v) for v in value]}
    if isinstance(value, tuple):
        return {
            "__sniff_type__": "tuple",
            "data": [_encode_meta_value(v) for v in value],
        }
    if isinstance(value, dict):
        return {str(k): _encode_meta_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_meta_value(v) for v in value]
    raise TypeError(f"not serialisable by SNIFF: {type(value)!r}")


def _decode_meta_value(value):
    """
    Reverse _encode_meta_value into original datatypes.
    Searches for "__sniff_type__" to check need for conversion.
    """
    if isinstance(value, dict):
        kind = value.get("__sniff_type__")
        if kind == "ndarray":
            return np.asarray(value.get("data"), dtype=value.get("dtype"))
        if kind == "dataframe":
            if isinstance(value.get("data"), str):
                return pd.read_json(StringIO(value["data"]), orient="split")
            index = _decode_pandas_index(value.get("index"))
            columns = [
                pd.Series(
                    [_decode_meta_value(v) for v in spec.get("values", [])],
                    index=index,
                    dtype=spec.get("dtype"),
                )
                for spec in value.get("frame", [])
            ]
            frame = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=index)
            frame.columns = _decode_pandas_index(value.get("columns"))
            return frame
        if kind == "series":
            if isinstance(value.get("data"), str):
                return pd.read_json(
                    StringIO(value["data"]), orient="split", typ="series"
                )
            return pd.Series(
                [_decode_meta_value(v) for v in value.get("values", [])],
                index=_decode_pandas_index(value.get("index")),
                dtype=value.get("dtype"),
                name=value.get("name"),
            )
        if kind == "path":
            return Path(value.get("data", ""))
        if kind == "set":
            return set(_decode_meta_value(v) for v in value.get("data", []))
        if kind == "tuple":
            return tuple(_decode_meta_value(v) for v in value.get("data", []))
        return {k: _decode_meta_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_meta_value(v) for v in value]
    return value


def _loads_meta(text) -> dict:
    """
    Decode JSON string back into a dict of restored objects. Runs through decode_meta_value.

    Parameters
    ----------
    text : str, bytes, or None
        The stored JSON text.

    Returns
    -------
    dict
        The decoded dictionary. Empty if undecodeable.
    """
    if text is None or text == "":
        return {}
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    try:
        result = json.loads(text)
    except (TypeError, ValueError):
        return {}
    if not isinstance(result, dict):
        return {}
    return {k: _decode_meta_value(v) for k, v in result.items()}


#############
# SAVE/LOAD #
#############


def _detach_from_memmap(stacks: List[Stack], path: Path) -> int:
    """
    Fully load stack data originally memory-mapped from path.
    Enables project overwrite.

    Returns
    -------
    int
        How many stacks were read into memory.
    """
    detached = 0
    try:
        target = path.resolve()
    except OSError:
        return detached
    for stack in stacks:
        data = getattr(stack, "data", None)
        if not isinstance(data, np.memmap):
            continue
        filename = getattr(data, "filename", None)
        if filename is None:
            continue
        try:
            same_file = Path(filename).resolve() == target
        except OSError:
            continue
        if same_file:
            stack.data = np.array(data)
            detached += 1
    return detached


def save_project(path, stacks: List[Stack], gui_state: Optional[dict]) -> Path:
    """
    Write project file to path. Writes to temporary name before renaming to avoid corruption.

    Parameters
    ----------
    path : str or Path
        Destination file path. ".sniff" is appended when the path has
        no suffix.
    stacks : List[Stack]
        Stacks to write.
    gui_state : dict, optional
        GUI field values to write.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".sniff")
    if _detach_from_memmap(stacks, path):
        gc.collect()  # garbage collector

    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        _write_project(tmp_path, stacks, gui_state)
        if path.exists():  # overwrite old file
            os.chmod(tmp_path, path.stat().st_mode & 0o7777)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return path


def _write_project(path: Path, stacks: List[Stack], gui_state: Optional[dict]) -> None:
    """Write the project's HDF5 content to path, which is overwritten."""
    with h5py.File(str(path), "w") as file:
        file.attrs["format"] = (
            "SNIFF_PROJECT"  # signature to ensure we are not loading the wrong h5 files.
        )

        ui = file.create_group("ui")
        ui.attrs["state"] = json.dumps(_encode_meta_value(gui_state or {}))

        sg = file.create_group("stacks")
        sg.attrs["count"] = len(stacks)

        for i, stack in enumerate(stacks):
            group = sg.create_group(f"stack_{i:05d}")
            group.attrs["path"] = "" if stack.path is None else str(stack.path)
            group.attrs["meta"] = json.dumps(
                _encode_meta_value(getattr(stack, "stack_meta", {}))
            )

            data = np.ascontiguousarray(
                np.asarray(stack.data, dtype=np.float32)
            )  # allows memmap on load
            if data.ndim == 2:  # 1-frame stack case
                data = data[None, :, :]
            group.create_dataset("data", data=data)

            infos = list(getattr(stack, "headers", None) or [])
            image_group = group.create_group("headers")
            image_group.create_dataset(
                "header",
                data=np.array(
                    [(x.tostring(padding=False) or "") for x in infos], dtype=object
                ),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )


def _read_headers(group) -> List[fits.Header]:
    """
    Reconstruct a stack's per-frame FITS headers from its stored group.

    The headers are written as one FITS header string per frame, in the
    'header' dataset of the stack's 'headers' group.

    Parameters
    ----------
    group : h5py.Group
        The stack's group.

    Returns
    -------
    List[fits.Header]
        One header per frame.
    """
    stored = group.get("headers")
    if stored is None:
        return []

    dataset = stored.get("header") if isinstance(stored, h5py.Group) else stored
    if dataset is None:
        return []

    return [
        fits.Header.fromstring(text) if text else fits.Header()
        for text in dataset.asstr()[()]
    ]


def _lazy_data(dataset, file_path) -> np.ndarray:
    """
    Return stack data as memmap so that data is loaded on access, allowing for low-RAM operation.
    Where memmap is not possible, fall back to fully loading array.
    """
    try:
        offset = dataset.id.get_offset()
        if offset is None:
            raise ValueError("dataset is not contiguously stored")
        return np.memmap(
            str(file_path),
            mode="r",
            dtype=dataset.dtype,
            shape=dataset.shape,
            offset=int(offset),
        )
    except Exception:
        return np.asarray(dataset[()])


def load_project(project_path, lazy: bool = True) -> Project:
    """
    Load a SNIFF project from file path.

    Parameters
    ----------
    project_path : str | Path
        Path to the SNIFF project file.
    lazy : bool
        Whether to read stacks via memory-map (if True, reduces RAM usage by only loading relevant Stack areas).

    Returns
    -------
    Project
        Populated Project class.
    """
    project_path = Path(project_path)
    file = h5py.File(str(project_path), "r")
    try:
        fmt = str(file.attrs.get("format", "") or "")
        if fmt and fmt != "SNIFF_PROJECT":
            raise ValueError(f"Not a SNIFF project file (format={fmt!r}).")

        stacks: List[Stack] = []
        if "stacks" in file:
            stack_group = file["stacks"]
            count = int(stack_group.attrs.get("count", 0))
            for i in range(count):
                key = f"stack_{i:05d}"
                if key not in stack_group:
                    continue
                group = stack_group[key]
                dataset = group["data"]
                data = _lazy_data(dataset, project_path) if lazy else np.asarray(dataset[()])
                headers = _read_headers(group)
                meta = _loads_meta(group.attrs.get("meta", ""))
                stack_path = str(group.attrs.get("path", "") or "")
                stacks.append(
                    Stack(
                        data=data,
                        headers=headers,
                        stack_meta=meta,
                        path=Path(stack_path) if stack_path else None,
                    )
                )

        ui_state = _loads_meta(file["ui"].attrs.get("state", "")) if "ui" in file else {}
    finally:
        file.close()

    return Project(stacks=stacks, gui_state=ui_state, _file=None)
