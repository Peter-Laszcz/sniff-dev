"""
Helper for extracting shutter times/ wavelengths from run folders.
"""

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


def extract_run_stats(dir: Path) -> dict[str, np.ndarray | None]:
    """
    Load the shutter/spectra statistics files found in a directory.

    Scans for spectra.txt, shuttertimes.txt and shuttercount.txt files and extracts the relevant column from each.

    Parameters
    ----------
    dir : Path
        Directory to scan for the txt files.

    Returns
    -------
    dict[str, np.ndarray | None]
        Dict of loaded arrays.
    """
    suffixes: Dict[str, Path | None] = {
        "spectra.txt": None,
        "shuttertimes.txt": None,
        "shuttercount.txt": None,
    }
    for suffix in suffixes.keys():
        for file in dir.iterdir():
            if file.is_file() and file.suffix.lower() == suffix:
                suffixes[suffix] = file
                break

    spec_path = suffixes["spectra.txt"]
    time_path = suffixes["shuttertimes.txt"]
    count_path = suffixes["shuttercount.txt"]

    spec_file = (
        pd.read_csv(spec_path, delimiter="\t", header=None)
        if spec_path is not None
        else None
    )
    shutter_time_file = (
        pd.read_csv(time_path, delimiter="\t", header=None)
        if time_path is not None
        else None
    )
    shutter_count_file = (
        pd.read_csv(count_path, delimiter="\t", header=None)
        if count_path is not None
        else None
    )
    shutters = (
        (shutter_time_file.iloc[:, 1] + shutter_time_file.iloc[:, 2]).to_numpy()
        if shutter_time_file is not None
        else None
    )
    shutter_count = (
        shutter_count_file.iloc[:, 1].to_numpy()
        if shutter_count_file is not None
        else None
    )
    spectra_time = spec_file.iloc[:, 0].to_numpy() if spec_file is not None else None

    return {
        "Shutter Times": shutters,
        "Shutter Count": shutter_count,
        "Spectral Times": spectra_time,
    }


def _first_shutter_count(
    overlap_data: Optional[Mapping[str, np.ndarray]],
) -> float | None:
    """
    Return the count of first shutter (used for normalisation).
    """
    counts = (overlap_data or {}).get("shutter_count")
    if counts is None:
        return None
    counts = np.asarray(counts)
    if counts.size == 0:
        return None
    first = (
        counts[0, 1] if counts.ndim > 1 and counts.shape[1] > 1 else counts.ravel()[0]
    )
    return float(first)


def frame_wavelengths(
    times: np.ndarray,
    delay: float,
    collimation_distance: float,
    apply_delay: bool = False,
) -> np.ndarray:
    """
    Convert per-frame times of flight into wavelengths.

    Neutron wavelength in angstroms is (3956 * ToF)/(flightpath length).

    Parameters
    ----------
    times : np.ndarray
        Per-frame times of flight, in seconds.
    delay : float
        Acquisition delay.
    collimation_distance : float
        Flightpath length, in metres.
    apply_delay : bool
        Whether to apply the delay.

    Returns
    -------
    np.ndarray
        One wavelength per time.
    """
    return (3956.0 / collimation_distance) * (times + (delay if apply_delay else 0.0))
