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
Scrubbing-correction weighting.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import pandas as pd

if TYPE_CHECKING:  # Stack sits above this module; annotation only
    from sni_app.core.components.stack import Stack

_WEIGHTS_COLUMNS = ["Folder", "w1", "w2", "OB1", "OB2"]
"""Columns of the weights table _weighting_func returns."""

TIMESTAMPS_FILE = "timestamps.txt"


_FOLDER_COLUMNS = ("folder", "subfolder", "directory", "dir", "name")
"""Column names accepted for the acquisition folder, normalised."""

_MODIFICATION_COLUMNS = ( #TODO: is there a better way to generalise?
    "modification (s)",
    "modification",
    "modification time",
    "modified",
    "time (s)",
    "time",
    "mtime"
)
"""Column names accepted for the modification time, normalised."""

_READ_ATTEMPTS = (
    {"sep": None, "engine": "python"},
    {"sep": ","},
    {"sep": "\t"},
    {"sep": r"\s+", "engine": "python"},
)
"""How a timestamps table is parsed, in order, until one yields the columns."""


def _find_nearest_lower_value(key, sorted_list):
    """
    Return the largest value in a list that is less than or equal to key, or smallest element.

    Parameters
    ----------
    key : float
        Reference value
    sorted_list : list
        Iterable of comparable values.

    Returns
    -------
    The nearest element less than or equal to key.
    """
    if key <= sorted(sorted_list)[0]:
        return sorted(sorted_list)[0]
    return max(i for i in sorted_list if i <= key)


def _find_nearest_upper_value(key, sorted_list):
    """
    Return the smallest value in list that is greater than or equal to key, or largest element.

    Parameters
    ----------
    key : float
        Reference value.
    sorted_list : list
        Iterable of comparable values.

    Returns
    -------
    The nearest element greater than or equal to key.
    """
    if key >= sorted(sorted_list)[-1]:
        return sorted(sorted_list)[-1]
    return min(i for i in sorted_list if i >= key)


def _merge_weights(existing, incoming):
    """
    Merge two scrubbing-weights dataframes, keeping the newest duplicate rows.

    Parameters
    ----------
    existing : pandas.DataFrame or None
        Previously accumulated weights.
    incoming : pandas.DataFrame or None
        Newly computed weights to fold in.

    Returns
    -------
    pandas.DataFrame or None
        Merged weights.
    """
    if incoming is None:
        return existing
    if existing is None:
        return incoming
    try:
        merged = pd.concat([existing, incoming], ignore_index=True)
        if "Folder" in merged.columns:
            merged = merged.drop_duplicates(subset="Folder", keep="last")
        else:
            merged = merged.drop_duplicates()
        return merged.reset_index(drop=True)
    except Exception:
        return incoming


def _as_path_list(folders: Optional[Union[str, Path, Iterable]]) -> List[Path]:
    """
    Normalise a folder selection into a list of paths.
    """
    if folders is None:
        return []
    if isinstance(folders, (str, Path)):
        folders = [folders]
    return [Path(str(folder)) for folder in folders if str(folder).strip()]


def _ob_folder_names(ob_folders: Optional[Union[str, Path, Iterable]]) -> List[str]:
    """
    Normalise an open-beam selection into a list of folder names.
    """
    return [folder.name for folder in _as_path_list(ob_folders)]


def _canonical_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce timestamp dataframe to desired data, i.e. folder & modification time.

    Parameters
    ----------
    df : pandas.DataFrame
        Timestamps table

    Returns
    -------
    pandas.DataFrame
        Reduced dataframe.

    Raises
    ------
    ValueError
        If either column is missing, or no row carries a usable time.
    """
    found: dict = {}
    for column in df.columns:
        key = " ".join(str(column).split()).lower()
        for canonical, accepted in (
            ("Folder", _FOLDER_COLUMNS),
            ("Modification (s)", _MODIFICATION_COLUMNS),
        ):
            if key in accepted and canonical not in found:
                found[canonical] = column

    missing = [name for name in ("Folder", "Modification (s)") if name not in found]
    if missing:
        raise ValueError(f"no {' or '.join(missing)} column (found {list(df.columns)})")

    table = df.rename(columns={column: name for name, column in found.items()})
    table["Folder"] = table["Folder"].astype(str).str.strip()
    table["Modification (s)"] = pd.to_numeric(
        table["Modification (s)"], errors="coerce"
    )
    table = table.dropna(subset=["Modification (s)"])
    if table.empty:
        raise ValueError("no row contains modification time")
    table = table.drop_duplicates(subset="Folder", keep="last")
    return table.sort_values(by="Modification (s)").reset_index(drop=True)


def read_timestamps_table(timestamps: Union[str, Path]) -> pd.DataFrame:
    """
    Read a timestamps table from a text file or containing folder.


    Parameters
    ----------
    timestamps : str or Path
        The table file, or a folder containing a 'timestamps.txt'.

    Returns
    -------
    pandas.DataFrame
        The table.

    Raises
    ------
    FileNotFoundError
        If no table is at that path.
    ValueError
        If the file cannot be parsed into the two required columns.
    """
    source = Path(str(timestamps))
    if source.is_dir():
        source = source / TIMESTAMPS_FILE
    if not source.is_file():
        raise FileNotFoundError(f"No timestamps table at '{source}'.")

    reason: Optional[Exception] = None
    for attempt in _READ_ATTEMPTS:
        try:
            parsed = pd.read_csv(
                source, skipinitialspace=True, keep_default_na=False, **attempt
            )
            return _canonical_timestamps(parsed)
        except Exception as exc:
            reason = reason or exc
    raise ValueError(f"Could not read the timestamps table '{source}': {reason}")


def _weighting_func(
    src: Optional[Path] = None,
    ob_folders: Optional[Union[str, Path, Iterable]] = None,
    timestamps: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Build the open-beam interpolation weights for every acquisition in a run.

    Reads/generates the timestamps.txt file in src, then calculates interpolated weights per acquisition using
    open beams.

    Parameters
    ----------
    src : Path, optional
        Experiment directory containing the acquisition subfolders (and, ideally,
        timestamps.txt). Required unless timestamps is given.
    ob_folders : str, Path or iterable, optional
        The folder(s) to treat as open beams. When
        omitted, folders whose name contains "ob" are used.
    timestamps : str or Path, optional
        A timestamps table to read instead of src's own, as a file or as a
        folder holding one. Given one, nothing is written to src.

    Returns
    -------
    pandas.DataFrame
        Columns ['Folder', 'w1', 'w2', 'OB1', 'OB2'], one row per folder in the
        run. Empty (with those columns) when no open-beam folders are present.

    Raises
    ------
    ValueError
        If neither src nor timestamps is given, or the table cannot be read.
    """
    if timestamps is None:
        if src is None:
            raise ValueError("Give either a run directory or a timestamps table.")
        timestamps = os.path.join(src, TIMESTAMPS_FILE)
        if not os.path.exists(timestamps):
            txt_timestamps(str(src), str(src))
    df = read_timestamps_table(timestamps)

    chosen = _ob_folder_names(ob_folders)
    if chosen:
        is_ob = df["Folder"].astype(str).isin(chosen)
    else:
        is_ob = df["Folder"].str.contains("ob", case=False, na=False)
    OB_df = df[is_ob]

    if OB_df.empty:
        return pd.DataFrame(columns=_WEIGHTS_COLUMNS)

    ob_times = OB_df["Modification (s)"].to_list()

    for i in df.index:
        t = df.at[i, "Modification (s)"]
        low = _find_nearest_lower_value(t, ob_times)
        high = _find_nearest_upper_value(t, ob_times)
        if low == high:
            df.at[i, "w1"] = 1
            df.at[i, "w2"] = 0
        else:
            df.at[i, "w1"] = (high - t) / (high - low)
            df.at[i, "w2"] = (t - low) / (high - low)
        df.at[i, "OB1"] = OB_df[OB_df["Modification (s)"] == low]["Folder"].values[0]
        df.at[i, "OB2"] = OB_df[OB_df["Modification (s)"] == high]["Folder"].values[0]

    return df[_WEIGHTS_COLUMNS]


def _keep_dir(stacks: List[Stack], dirs: List[Path]) -> List[Stack]:
    """
    Filter a list of stacks down to those originating from given directories.

    Parameters
    ----------
    stacks : List[Stack]
        Stacks to filter.
    dirs : List[Path]
        Folder paths to keep. A stack is retained when its path is in
        this list.

    Returns
    -------
    List[Stack]
        The subset of stacks within the keep directories.
    """
    out = []
    for stack in stacks:
        if stack.path in dirs:
            out.append(stack)
    return out


def _keep_key_weights(
    stacks: List[Stack],
    weights_df: Optional[pd.DataFrame],
    keep_folder: Optional[Union[str, Path, Iterable]],
) -> List[Stack]:
    """
    Keep the requested folders together with their linked open-beam folders.

    Parameters
    ----------
    stacks : List[Stack]
        Stacks to filter.
    weights_df : pandas.DataFrame | None
        Scrubbing weights, as returned by :func:`_weighting_func`.
    keep_folder : str | Path | list | None
        Folder(s) requested for processing, by path. None is treated as an
        empty selection.

    Returns
    -------
    List[Stack]
        The requested stacks plus those of their linked open beams.
    """
    keep_paths = _as_path_list(keep_folder)
    if not keep_paths:
        return []

    if weights_df is None or getattr(weights_df, "empty", True):  # no weights
        return _keep_dir(stacks, keep_paths)

    # Ensure required columns exist
    required = {"Folder", "OB1", "OB2"}
    if not required.issubset(set(weights_df.columns)):
        return _keep_dir(stacks, keep_paths)

    keep_names = {path.name for path in keep_paths}
    rows = weights_df[weights_df["Folder"].astype(str).isin(keep_names)]
    ob_names = {
        str(name)
        for name in pd.concat([rows["OB1"], rows["OB2"]])
        if pd.notna(name) and str(name).strip()
    }
    roots = {path.parent for path in keep_paths}
    ob_paths = [root / name for root in roots for name in ob_names]

    return _keep_dir(stacks, keep_paths + ob_paths)


def txt_timestamps(src_dir: Path, dst_dir: Path):
    """
    Create or update a 'timestamps.txt' file containing median file timestamps per subfolder.

    Parameters
    ----------
    src_dir : str
        Source directory containing experiment subfolders.
    dst_dir : str
        Destination directory where 'timestamps.txt' will be written.

    Returns
    -------
    None
    """
    txt_file = os.path.join(dst_dir, TIMESTAMPS_FILE)

    write_header = not os.path.exists(txt_file) or os.path.getsize(txt_file) == 0

    subfolders = os.listdir(src_dir)
    subfolders.append(".")
    subfolders.sort()
    records = []

    for name in subfolders:
        subfolder = os.path.join(src_dir, name)

        # Ignore non-directories
        if not os.path.isdir(subfolder):
            continue

        files = sorted(os.listdir(subfolder))
        if not files:
            continue

        mid_file = files[len(files) // 2]
        mid_path = os.path.join(subfolder, mid_file)

        mod_sec = os.path.getmtime(mid_path)
        cre_sec = os.path.getctime(mid_path)

        records.append(
            [
                name,
                f"{mod_sec:.2f}",
                f"{cre_sec:.2f}",
                time.ctime(mod_sec),
                time.ctime(cre_sec),
            ]
        )

    df = pd.DataFrame(
        records,
        columns=[
            "Folder",
            "Modification (s)",
            "Creation (s)",
            "Formatted modification",
            "Formatted creation",
        ],
    )

    with open(txt_file, "a", newline="") as f:
        df.to_csv(f, header=write_header, index=False)
