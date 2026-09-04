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
Stack discovery and I/O.
"""

from __future__ import annotations

import os
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sni_app.core.util.logger as logger
from sni_app.core.io.image import ALLOWED_EXTENSIONS
from sni_app.core.util.scrubbing import _weighting_func

if TYPE_CHECKING:  # Stack sits above this module; see discover_and_load
    from sni_app.core.components.stack import Stack

_log = logger.setup_logger()

_RUN_META_SUFFIXES = {
    "shutter_count": "_ShutterCount.txt",
    "shutter_times": "_ShutterTimes.txt",
    "spectra": "_Spectra.txt",
}
"""
Filename suffixes of the experiment metadata text files a stack folder is
scanned for (IMAT and PSI standard), keyed by the role they fill.
"""


def scan_experiment_txts(stack_dir: Path) -> Dict[str, np.ndarray]:
    """
    Scan stack_dir for the experiment metadata text files and load them.

    Looks for file suffixes in STACK_META_SUFFIXES and returns a dict mapping each file type
    to array loaded from corresponding file. Returns empty fields if not files found.

    Parameters
    ----------
    stack_dir : Path
        Directory to scan for the experiment metadata text files.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of array type ("shutter_count", "shutter_times", "spectra")
        to the loaded array, containing only the roles found.
    """
    found: Dict[str, np.ndarray] = {}
    if not stack_dir.is_dir():
        return found
    for role, suffix in _RUN_META_SUFFIXES.items():
        matches = sorted(stack_dir.glob(f"*{suffix}"))
        if not matches:
            continue
        try:
            found[role] = pd.read_csv(matches[0], sep="\t", header=None).to_numpy()
        except Exception as exc:
            _log.warning(f"Could not read experiment metadata file {matches[0]}: {exc}")
    return found


def resolve_run_meta_array(
    stack_meta: Optional[dict],
    meta_type: str,
    path: str = "",
) -> np.ndarray:
    """
    Return from a given file the experiment metadata array of a stack for a given metadata type.
    If no path provided, uses stored arrays from stack folder (if applicable).

    Parameters
    ----------
    stack_meta : dict or None
        Stack's stack_meta dict.
    meta_type : str
        "shutter_count", "shutter_times" or "spectra".
    path : str, optional
        Path to a text file containing metadata array.

    Returns
    -------
    np.ndarray
        Metadata array for given type.

    Raises
    ------
    ValueError
        If neither a file nor internal data is available.
    """
    if path:
        return pd.read_csv(path, sep="\t", header=None).to_numpy()

    internal = (stack_meta or {}).get("run_meta") or {}
    if meta_type in internal:
        return internal[meta_type]

    raise ValueError("No file selected and no internal correction data.")


def _safe_file_stem(name: str) -> str:
    """
    Sanitise an arbitrary display name into a filesystem-safe stem.

    Parameters
    ----------
    name : str
        The name to sanitise.

    Returns
    -------
    str
        The name with disallowed characters replaced by "_" (or "stack"
        if string ends up empty).
    """
    keep = "-_.() "  # allowed characters (aside from alphanumeric)
    s = "".join(c if (c.isalnum() or c in keep) else "_" for c in str(name)).strip()
    return s or "stack"


def export_stacks(
    pairs: List[tuple],
    stem: str,
    ext: str,
    base_dir: Path,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, List[str]]:
    """
    Write a batch of stacks to disk, stack-by-stack.

    Parameters
    ----------
    pairs : List[tuple]
        List of stacks to export, in form (stack_name, stack).
    stem : str
        Filename prefix for the exported file(s).
    ext : str
        Output extension.
    base_dir : Path
        Destination directory.
    overwrite : bool, optional
        Whether to overwrite existing files.
    progress_callback : Callable[[int, int], None], optional
        Callback for progress bar.

    Returns
    -------
    Tuple[int, List[str]]
        The number of stacks written and a list of error strings.
    """
    total = len(pairs)
    written = 0
    errors: List[str] = []
    for idx, (name, stack) in enumerate(pairs, 1):
        file_name = (
            f"{stem}_{_safe_file_stem(name)}{ext}" if total > 1 else f"{stem}{ext}"
        )
        try:
            stack.save_stack(file_name, base_dir, overwrite)
            written += 1
        except Exception as exc:
            errors.append(f"'{name}': {exc}")
        if progress_callback is not None:
            progress_callback(idx, total)
    return written, errors


def discover_stack_dirs(
    parent: Path,
    logs: Optional[Logger] = None,
) -> List[Path]:
    """
    Recursively search for valid stack directories under a root.
    A valid directory needs at least one supported image.

    Parameters
    ----------
    parent : Path
        Base directory for search.
    logs : logging.Logger, optional
        Logger (if applicable) to report discovered directories and errors to.

    Returns
    -------
    List[Path]
        List of valid stack dirs.
    """
    entries: List[Path] = []
    parent_flag = False

    if not parent.is_dir():
        if logs is not None:
            logs.error(f"Stack root missing or not a directory: {parent}")
        return entries
    for dirpath, _dirnames, filenames in os.walk(parent):
        has_images = any(
            os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS
            for name in filenames
        )
        if not has_images:
            continue
        parent_flag = True
        path = Path(dirpath)
        if path == parent:
            continue
        entries.append(path)
        if logs is not None:
            logs.info(f"Found stack directory: {path}")
    if parent_flag:
        entries.append(parent)
        if logs is not None:
            logs.info(f"Found stack directory: {parent}")
    entries = list(
        set(entries)
    )  # TODO: This is an awful bodge to remove duplicates. search needs rethinking.
    if logs is not None:
        logs.info(f"Found {len(entries)} stack directories")
    return entries


def list_stack_frames(stack_dir: Path) -> List[Path]:
    """
    List the readable image frames in a stack directory, sorted by name.

    Parameters
    ----------
    stack_dir : Path
        Directory to list frames from.

    Returns
    -------
    List[Path]
        Sorted paths of the files with an allowed image extension; an empty list
        when none are present.
    """
    with os.scandir(stack_dir) as item:
        names = [
            entry.name
            for entry in item
            if entry.is_file()
            and os.path.splitext(entry.name)[1].lower() in ALLOWED_EXTENSIONS
        ]
    return [
        stack_dir / name for name in sorted(names, key=os.path.normcase)
    ]  # faster than sorting paths


###########
# READING #
###########


def discover_and_load(
    src_dir: Path,
    proc_folders: Optional[List[Path]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Stack], dict]:
    """
    Read every stack folder under a source directory.

    Discovers stack folders under src_dir, builds stacks for each
    folder containing readable images, and computes the scrubbing weights
    dataframe for the directory.  When proc_folders is provided,the stack
    list is filtered to those folders (plus their linked open-beam folders).

    Parameters
    ---------
        src_dir : Path
            Directory containing stack folders
        proc_folders: Optional[List[Path]] (default: None)
            List of folders to process
        progress_callback : Callable[[int, int], None], optional
            Progress bar interface.

    Returns
    -------
    Tuple[List[Stack], dict]
        The discovered stacks and a parameter dict including the weights dataframe.
    """
    # deferred: both live above this module and import it in turn
    from sni_app.core.components.stack import Stack
    from sni_app.core.util.scrubbing import _keep_key_weights

    stack_dirs = discover_stack_dirs(src_dir)
    frame_counts = {path: len(list_stack_frames(path)) for path in stack_dirs}
    total_frames = sum(frame_counts.values())
    frames_done = 0

    try:
        weights_data_frame = _weighting_func(src_dir)
    except Exception as exc:
        _log.warning(f"Could not build weights dataframe for {src_dir}: {exc}")
        weights_data_frame = None

    stacks: List[Stack] = []
    for path in stack_dirs:
        if progress_callback is not None:
            base = frames_done
            folder_callback = lambda done, _n, _base=base: progress_callback(
                _base + done, total_frames
            )
        else:
            folder_callback = None
        try:
            stack = Stack.from_folder(path, progress_callback=folder_callback)
            stack.stack_meta["weights_data_frame"] = weights_data_frame
            stacks.append(stack)
        except Exception as exc:  # Skip folders that hold no readable stack
            _log.warning(f"Could not read stack folder {path}: {exc}")
            continue
        finally:
            frames_done += frame_counts[path]
            if progress_callback is not None:
                progress_callback(frames_done, total_frames)

    if proc_folders:
        stacks = _keep_key_weights(stacks, weights_data_frame, proc_folders)

    param_dict = {}
    return stacks, param_dict
