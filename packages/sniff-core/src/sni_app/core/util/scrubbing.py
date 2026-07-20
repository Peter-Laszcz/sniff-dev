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

WEIGHTS_COLUMNS = ["Folder", "w1", "w2", "OB1", "OB2"]
"""Columns of the weights table weighting_func returns."""


def find_nearest_lower_value(key, sorted_list):
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


def find_nearest_upper_value(key, sorted_list):
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


def merge_weights(existing, incoming):
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


def weighting_func(
    src: Path, ob_folders: Optional[Union[str, Path, Iterable]] = None
) -> pd.DataFrame:
    """
    Build the open-beam interpolation weights for every acquisition in a run.

    Reads/generates the timestamps.txt file in src, then calculates interpolated weights per acquisition using
    open beams.

    Parameters
    ----------
    src : Path
        Experiment directory containing the acquisition subfolders (and, ideally,
        timestamps.txt).
    ob_folders : str, Path or iterable, optional
        The folder(s) to treat as open beams. When
        omitted, folders whose name contains "ob" are used.

    Returns
    -------
    pandas.DataFrame
        Columns ['Folder', 'w1', 'w2', 'OB1', 'OB2'], one row per folder in the
        run. Empty (with those columns) when no open-beam folders are present.
    """
    timestamps = os.path.join(src, "timestamps.txt")
    if not os.path.exists(timestamps):
        txt_timestamps(str(src), str(src))
    df = (
        pd.read_csv(timestamps)
        .sort_values(by="Modification (s)")
        .reset_index(drop=True)
    )

    chosen = _ob_folder_names(ob_folders)
    if chosen:
        is_ob = df["Folder"].astype(str).isin(chosen)
    else:
        is_ob = df["Folder"].str.contains("ob", case=False)
    OB_df = df[is_ob]

    if OB_df.empty:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    ob_times = OB_df["Modification (s)"].to_list()

    for i in df.index:
        t = df.at[i, "Modification (s)"]
        low = find_nearest_lower_value(t, ob_times)
        high = find_nearest_upper_value(t, ob_times)
        if low == high:
            df.at[i, "w1"] = 1
            df.at[i, "w2"] = 0
        else:
            df.at[i, "w1"] = (high - t) / (high - low)
            df.at[i, "w2"] = (t - low) / (high - low)
        df.at[i, "OB1"] = OB_df[OB_df["Modification (s)"] == low]["Folder"].values[0]
        df.at[i, "OB2"] = OB_df[OB_df["Modification (s)"] == high]["Folder"].values[0]

    return df[WEIGHTS_COLUMNS]


def keep_dir(stacks: List[Stack], dirs: List[Path]) -> List[Stack]:
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


def keep_key_weights(
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
        Scrubbing weights, as returned by :func:`weighting_func`.
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
        return keep_dir(stacks, keep_paths)

    # Ensure required columns exist
    required = {"Folder", "OB1", "OB2"}
    if not required.issubset(set(weights_df.columns)):
        return keep_dir(stacks, keep_paths)

    keep_names = {path.name for path in keep_paths}
    rows = weights_df[weights_df["Folder"].astype(str).isin(keep_names)]
    ob_names = {
        str(name)
        for name in pd.concat([rows["OB1"], rows["OB2"]])
        if pd.notna(name) and str(name).strip()
    }
    roots = {path.parent for path in keep_paths}
    ob_paths = [root / name for root in roots for name in ob_names]

    return keep_dir(stacks, keep_paths + ob_paths)


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
    txt_file = os.path.join(dst_dir, "timestamps.txt")

    if not os.path.exists(txt_file):
        Path(txt_file).touch()
        with open(txt_file, "a") as f:
            f.write(
                "Folder,Modification (s),Creation (s),Formatted modification,Formatted creation\n"
            )

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

    with open(txt_file, "a") as f:
        df.to_csv(f, header=False, index=False)
