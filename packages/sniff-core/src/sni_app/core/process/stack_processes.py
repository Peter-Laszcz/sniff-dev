"""
Stack-level processing operations.

Stack functions receive a stack or list of stacks and outputs processed stacks or images, applying process
history where appropriate.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from astropy.io import fits

from sni_app.core.components.stack import (
    Stack,
    record_derivation,
)
from sni_app.core.io.stack import resolve_run_meta_array
from sni_app.core.process.img_processes import (
    BlackBodyFit,
    extract_features,
    normalise_frame,
    register_frame_to_features,
)

_log = logging.getLogger("SNIFF_Log")

OVERLAP_ROLES = ("shutter_count", "shutter_times", "spectra")
"""
Roles of the experiment run tables consumed by the overlap correction, in the
order compute_shutter_indices expects them.
"""


def _kept_spectra_times(
    stack: Stack, frames: Optional[Sequence[int]] = None
) -> Optional[np.ndarray]:
    """
    Per-frame acquisition times of the frames a process kept from a stack.

    Processes that change the frame count call this with the indices of the
    input frames their output is built from, so the times follow the data (they
    live under stack_meta["spectra_times"], and are read back with
    Stack.times_of_flight). Returns None when the stack has no times, or when it has
    too few to describe the frames asked for.

    Parameters
    ----------
    stack : Stack
        The input stack the frames were taken from.
    frames : sequence of int, optional
        Indices into stack's frames, in output order. Defaults to every frame.

    Returns
    -------
    np.ndarray | None
        The times of those frames, or None when they cannot be derived. An
        empty selection of a stack that has times gives an empty array.
    """
    times = stack.times_of_flight()
    if times is None:
        return None
    if frames is None:
        frames = range(int(stack.data.shape[0]))
    indices = np.asarray(list(frames), dtype=int)
    if indices.max(initial=-1) >= times.size:
        return None
    return times[indices]


def _reduced_spectra_times(
    stacks: Sequence[Stack], n_frames: int
) -> Optional[np.ndarray]:
    """
    Per-frame times for a process combining whole stacks.

    The inputs should be acquisitions of the same frames, so the first stack that
    carries times (and has one per output frame) dates the result.

    Parameters
    ----------
    stacks : sequence of Stack
        The stacks the process combined.
    n_frames : int
        Frame count of the stack produced.

    Returns
    -------
    np.ndarray | None
        The times, or None when no input supplies a matching set.
    """
    for stack in stacks:
        times = stack.times_of_flight()
        if times is not None and times.size == int(n_frames):
            return times
    return None


def stack_avg(stacks: list[Stack], **kwargs) -> List[Stack]:
    """
    Average multiple stacks into a single mean stack via elementwise nanmean.
    Resultant stack is populated with median header.

    Parameters
    ----------
    stacks : list[Stack]
        Stacks to average together.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        A single-element list storing the averaged stack.
    """
    list_stacks = [stack.data for stack in stacks]
    list_headers = []
    for x in [stack.headers for stack in stacks]:
        list_headers.extend(x)

    stack_mean = np.nanmean(list_stacks, axis=0)
    median_info: fits.Header = list_headers[len(list_headers) // 2]
    median_info.update({"HISTORY": "Sequence: Stack average. X-axis (acquisitions)"})

    info = [median_info] * (np.shape(stack_mean)[0])

    # Averaging is frame-for-frame, so the first stack's times still describe
    # the result.
    meta = {"spectra_times": (_reduced_spectra_times(stacks, stack_mean.shape[0]))}
    return record_derivation(
        [Stack(stack_mean, info, meta, None)],
        "Stack Averaging",
        {},
        stacks,
        mode="reduce",
    )


def stack_sum(stacks: list[Stack], **kwargs) -> List[Stack]:
    """
    Sum multiple stacks into a single stack (element-wise) using nansum.
    Resultant stack is populated with median header.

    Parameters
    ----------
    stacks : list[Stack]
        Stacks to sum together. Must be same shape.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        A single-element list holding the summed stack.

    Raises
    ------
    ValueError
        If stacks is empty or the stacks do not all share the same shape.
    """
    if not stacks:
        raise ValueError("No stacks to sum.")
    shapes = {tuple(stack.data.shape) for stack in stacks}
    if len(shapes) > 1:
        raise ValueError(f"Cannot sum stacks with differing shapes: {sorted(shapes)}")

    list_stacks = [stack.data for stack in stacks]
    list_infos = []
    for stack in stacks:
        list_infos += stack.headers

    stack_summed = np.nansum(list_stacks, axis=0)
    median_headers = list_infos[len(list_infos) // 2]
    median_headers.update({"HISTORY": "Sequence: Stack summation"})

    headers = [median_headers] * (np.shape(stack_summed)[0])

    meta = {"spectra_times": (_reduced_spectra_times(stacks, stack_summed.shape[0]))}
    return record_derivation(
        [Stack(stack_summed, headers, meta, None)],
        "Stack Summation",
        {},
        stacks,
        mode="reduce",
    )


def separate_energies(stack, he_slice, le_slice):
    """
    Collapse a stack into a two-frame high-energy / low-energy stack.
    Averages energy bands using nanmean and median header.

    Parameters
    ----------
    stack : Stack
        Source stack to split by energy.
    he_slice : tuple[int, int]
        (start, stop) frame indices defining the high-energy band.
    le_slice : tuple[int, int]
        (start, stop) frame indices defining the low-energy band.

    Returns
    -------
    Stack
        A two-frame stack: containing high_energy_avg and low_energy_avg.
    """
    he_avg = np.nanmean(stack.data[he_slice[0] : he_slice[1], :, :], axis=0)
    le_avg = np.nanmean(stack.data[le_slice[0] : le_slice[1], :, :], axis=0)
    he_hdr = stack.headers[he_slice[0] + ((he_slice[1] - he_slice[0]) // 2)]
    le_hdr = stack.headers[le_slice[0] + ((le_slice[1] - le_slice[0]) // 2)]
    he_hdr.update({"HISTORY": f"Sequence: Separate energies. HE: {he_slice}"}),
    le_hdr.update({"HISTORY": f"Sequence: Separate energies. LE: {le_slice}"})
    return Stack(
        data=np.stack([he_avg, le_avg], axis=0),
        headers=([he_hdr, le_hdr]),
        stack_meta={},
        path=None,
    )


def stack_slice_acquisitions(
    stacks: List[Stack], start=0, stop=None, **kwargs
) -> List[Stack]:
    """
    Slice stack framewise.

    Parameters
    ----------
    stacks : List[Stack]
        Stacks to slice.
    start : int, optional
        First frame index to keep (default 0).
    stop : int or None, optional
        Stop index (exclusive); None (default) keeps through the last frame.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        Sliced stack.
    """
    out = []
    for stack in stacks:
        sliced = stack.data[start:stop, :, :]
        headers = []
        for img in stack.headers[start:stop]:
            img.update(
                {
                    "HISTORY": f"Sequence: Sliced acquisition frames. Start={start}, Stop={stop}"
                }
            )
            headers.append(img)
        kept = range(*slice(start, stop).indices(int(stack.data.shape[0])))
        meta = {"spectra_times": (_kept_spectra_times(stack, kept))}
        out.append(Stack(sliced, headers, stack_meta=meta, path=None))
    return record_derivation(
        out, "Stack Slicer", {"start": start, "stop": stop}, stacks, mode="map"
    )


def stack_bin_frames(
    stacks: List[Stack],
    bin_factor: int = 1,
    start_img: int = 0,
    he_le: tuple[bool, list, list] | None = (False, [15, 30], [30, 50]),
    **kwargs,
) -> List[Stack]:
    """
    Bin consecutive frames within each stack by averaging. bin_factor decides bin size.
    Alternatively, if he_le is enabled, split into a high/low energy pair.

    Parameters
    ----------
    stacks : List[Stack]
        Stacks whose frames are binned.
    bin_factor : int, optional
        Number of frames averaged per output frame (default 1).
    start_img : int, optional
        First frame index to bin from (default 0).
    he_le : tuple, optional
        (enabled, he_range, le_range). When enabled is true the stack
        is separated into high-/low-energy frames instead of binned. Default
        (False, [15, 30], [30, 50]).
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        The binned (or energy-separated) stacks.
    """
    out = []
    for stack in stacks:
        if he_le:
            out.append(separate_energies(stack, he_slice=he_le[1], le_slice=he_le[2]))
            continue
        num_bins = (len(stack.headers) - start_img) // bin_factor
        if num_bins < 1:
            raise ValueError(
                f"No complete bins from frame {start_img} with bin factor {bin_factor}"
            )
        binned = stack.data[start_img : start_img + num_bins * bin_factor, :, :]
        binned = binned.reshape(num_bins, bin_factor, *binned.shape[1:]).mean(axis=1)
        # Each output frame is dated (like its header) by the middle frame of
        # the bin it averages.
        centres = [
            start_img + i * bin_factor + (bin_factor // 2) for i in range(num_bins)
        ]
        headers = []
        for i, centre in enumerate(centres):
            stack.headers[centre].update(
                {
                    "HISTORY": f"Sequence: Frames binning. Binning: {bin_factor}. Starting image: {start_img + i * bin_factor}"
                }
            )
            headers.append(stack.headers[centre])

        meta = {"spectra_times": (_kept_spectra_times(stack, centres))}
        out.append(Stack(data=binned, headers=headers, stack_meta=meta, path=None))
    return record_derivation(
        out,
        "Bin Stack Frames",
        {"bin_factor": bin_factor, "start_img": start_img, "he_le": he_le},
        stacks,
        mode="map",
    )


def stack_join(stacks: List[Stack], **kwargs) -> List[Stack]:
    """
    Concatenate stacks in given order. Frames must be same shape.

    Parameters
    ----------
    stacks : List[Stack]
        Stacks to join, in order of concatenation.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        List holding concatenated stack.

    Raises
    ------
    ValueError
        If stacks is empty or the stacks' frame shapes differ.
    """
    if not stacks:
        raise ValueError("No stacks to join.")
    shapes = {tuple(stack.data.shape[-2:]) for stack in stacks}
    if len(shapes) > 1:
        raise ValueError(
            f"Cannot join stacks with differing frame shapes: {sorted(shapes)}"
        )

    data = np.concatenate([np.asarray(stack.data) for stack in stacks], axis=0)
    headers = []
    for pos, stack in enumerate(stacks, 1):
        note = f"Sequence: Join stacks. Source stack {pos}/{len(stacks)}"
        for img in stack.headers:
            img.update({"HISTORY": note})
            headers.append(img)

    # The joined times are only meaningful if every source has some.
    times = [_kept_spectra_times(stack) for stack in stacks]
    meta = {
        "spectra_times": (
            np.concatenate(times) if all(t is not None for t in times) else None
        )
    }
    return record_derivation(
        [Stack(data=data, headers=headers, stack_meta=meta, path=None)],
        "Join Stacks",
        {"order": [stack.display_name() for stack in stacks]},
        stacks,
        mode="reduce",
    )


def stack_scrubbing(
    stacks: List[Stack],
    weights: Optional[pd.DataFrame] = None,
    open_beam_dir: Union[str, Path, None] = None,
    **kwargs,
) -> List[Stack]:
    """
    Apply scrubbing correction per stack.

    By default the weights dataframe supplies, per stack, the two open-beam
    folders (OB1/OB2) and their interpolation weights (w1/w2); every frame is
    divided by the weighted mean of the two open-beam averages, and stacks with
    no matching weights row are skipped.
    Passing open_beam_dir overrides the given values in the weights dataframe.

    Parameters
    ----------
    stacks : List[Stack]
        Sample stacks to correct.
    weights : pandas.DataFrame, optional
        Scrubbing weights, with columns:
        Folder, w1, w2, OB1, OB2. Required unless open_beam_dir is given.
    open_beam_dir : str or Path, optional
        Path to a single open-beam folder to correct every stack against.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        Corrected stacks.

    Raises
    ------
    ValueError
        If neither weights nor open_beam_dir is supplied.
    """
    if open_beam_dir in ("", None) and weights is None:
        raise ValueError(
            "Scrubbing correction needs either a weights dataframe or an "
            "open-beam folder."
        )

    ob_avg_cache: dict = {}

    def ob_avg(folder: Path):
        key = str(folder)
        if key not in ob_avg_cache:
            ob_avg_cache[key] = np.nanmean(Stack.from_folder(Path(folder)).data, axis=0)
        return ob_avg_cache[key]

    chosen_ob = Path(open_beam_dir) if open_beam_dir else None

    out = []
    corrected_from = []
    for stack in stacks:
        if chosen_ob is not None:
            w1, w2 = 1.0, 0.0
            ob1_dir = ob2_dir = chosen_ob
        else:
            if stack.path is None:
                continue
            stack_path = Path(stack.path)
            row = weights[weights["Folder"].astype(str) == stack_path.name]
            if row.empty:
                continue

            w1 = row["w1"].values[0]  # weights
            w2 = row["w2"].values[0]
            ob1_dir = stack_path.parent / str(row["OB1"].values[0])
            ob2_dir = stack_path.parent / str(row["OB2"].values[0])

        data = stack.data / (ob_avg(ob1_dir) * w1 + ob_avg(ob2_dir) * w2)
        headers = []
        for header in stack.headers:
            header.update({"HISTORY": f"Scrubbing correction: w1={w1}, w2={w2}"})
            headers.append(header)

        # Copy the source metadata as sharing the dict would let the lineage
        # recorded below overwrite the input stack's id and history.
        out.append(
            Stack(
                data=data,
                headers=headers,
                stack_meta=dict(stack.stack_meta or {}),
                path=None,
            )
        )
        corrected_from.append(stack)
    params = {"open_beam_dir": str(chosen_ob)} if chosen_ob is not None else {}
    return record_derivation(
        out, "Scrubbing Correction", params, corrected_from, mode="map"
    )


def stack_sbkg_correction(
    stacks: List[Stack],
    bb_mask: Stack,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> List[Stack]:
    """
    Perform black-body correction.

    Parameters
    ----------
    stacks : List[Stack]
        Stacks to correct.
    bb_mask : Stack
        The black-body mask, as a single-image stack. Any non-zero pixel is
        taken as masked.
    progress_callback : Callable[[int, int], None], optional
        Called with (frames done, frames total) as the correction proceeds.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        One corrected stack per input, in order.

    Raises
    ------
    ValueError
        If the mask is not a single image, marks too few black bodies, or does
        not match a stack's frame shape.
    """
    mask = np.asarray(bb_mask.data)
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError(
                f"Black-body mask must be a single image, got {mask.shape[0]} frames."
            )
        mask = mask[0]

    # The mask is the same for every frame, so the fit is prepared once.
    fit = BlackBodyFit(mask)
    note = f"SBKG correction: {len(fit)} black bodies"

    # Shapes are checked up front: a mismatch on the last stack should not cost
    # every fit before it.
    total = 0
    for stack in stacks:
        if stack.data.shape[-2:] != fit.shape:
            raise ValueError(
                f"Black-body mask {fit.shape} does not match frame shape "
                f"{stack.data.shape[-2:]}."
            )
        total += int(stack.data.shape[0])

    done = 0
    out = []
    for stack in stacks:
        headers = [header.copy() for header in stack.headers]
        for header in headers:
            header.update({"HISTORY": note})

        corrected = np.empty_like(stack.data, dtype=np.float32)
        for index, frame in enumerate(stack.data):
            corrected[index] = frame - fit.background(frame)
            if progress_callback is not None:
                progress_callback(done + index + 1, total)
        done += len(corrected)

        out.append(Stack.from_array(corrected, headers, stack.stack_meta))

    return record_derivation(
        out,
        "SBKG Correction",
        {},
        stacks,
        aux={"bb_mask": bb_mask},
        mode="map",
    )


def stack_referencing(stacks: List[Stack], ref: np.ndarray) -> List[Stack]:
    """
    Divide every stack by a common reference image.

    Parameters
    ----------
    stacks : List[Stack]
        Stacks to reference.
    ref : np.ndarray
        Reference image to divide by.

    Returns
    -------
    List[Stack]
        Divided stacks.
    """
    out = []
    for stack in stacks:
        headers = []
        for header in stack.headers:
            header.update({"HISTORY": "Referencing correction"})
            headers.append(header)
        out.append(
            Stack(
                data=np.divide(stack.data, ref),
                headers=headers,
                stack_meta=dict(stack.stack_meta or {}),
                path=None,
            )
        )
    return out


def overlap_correct_array(
    data: np.ndarray,
    shutter_count: np.ndarray,
    shutter_times: np.ndarray,
    spectra: np.ndarray,
) -> np.ndarray:
    """
    Run overlap correction on a raw stack array.

    Parameters
    ----------
    data : ndarray
        Sample stack data from experiment.
    shutter_count : ndarray
        Experiment _ShutterCount.txt loaded as numpy array using Pandas (tab-delimited)
    shutter_times : ndarray
        Experiment _ShutterTimes.txt loaded as numpy array using Pandas (tab-delimited)
    spectra : ndarray
        Experiment _Spectra.txt loaded as numpy array using Pandas (tab-delimited)
    Returns
    -------
    ndarray
        Overlap-corrected array.
    """
    if data.shape[0] == (spectra.shape[0] + 1):
        data = data[:-1]

    start_indices, end_indices, counts = compute_shutter_indices(
        shutter_count, shutter_times, spectra
    )

    prob_occupied = np.zeros_like(data, dtype=np.float32)
    for ss, se, count in zip(start_indices, end_indices, counts, strict=True):
        np.cumsum(data[ss : se - 1], axis=0, out=prob_occupied[ss + 1 : se])
        prob_occupied[ss + 1 : se] /= count

    np.subtract(1, prob_occupied, out=prob_occupied)
    return data / prob_occupied


def stack_overlap_correction(
    stacks: List[Stack],
    shutter_count: str = "",
    shutter_times: str = "",
    spectra: str = "",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> List[Stack]:
    """
    Overlap-correct each stack against its experiment metadata arrays.

    Parameters
    ----------
    stacks : List[Stack]
        Sample stacks from the experiment.
    shutter_count, shutter_times, spectra : str, optional
        Paths to tab-delimited run tables overriding the stacks' internal
        arrays. Empty (the default) uses each stack's own data.
    progress_callback : Callable[[int, int], None], optional
        Called with (stacks done, total) after each stack, if provided.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        Overlap-corrected stacks.

    Raises
    ------
    ValueError
        If a stack has neither an override path nor internal correction data.
    """
    overrides = {
        "shutter_count": shutter_count,
        "shutter_times": shutter_times,
        "spectra": spectra,
    }
    out = []
    for index, stack in enumerate(stacks, 1):
        arrays = [
            resolve_run_meta_array(stack.stack_meta, role, overrides[role])
            for role in OVERLAP_ROLES
        ]
        corrected = overlap_correct_array(np.asarray(stack.data), *arrays)
        headers = stack.headers[: corrected.shape[0]]
        out.append(Stack.from_array(corrected, headers, stack.stack_meta))
        if progress_callback is not None:
            progress_callback(index, len(stacks))
    return record_derivation(
        out,
        "Overlap Correction",
        {role: (path or "(internal)") for role, path in overrides.items()},
        stacks,
        mode="map",
    )


def compute_shutter_indices(
    shutter_count: np.ndarray,
    shutter_times: np.ndarray,
    spectra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produces shutter start and end indices (and counts).

    Parameters
    ----------
    shutter_count : ndarray
        Experiment _ShutterCount.txt loaded as numpy array using Pandas (tab-delimited).
    shutter_times : ndarray
        Experiment _ShutterTimes.txt loaded as numpy array using Pandas (tab-delimited).
    spectra : ndarray
        Experiment _Spectra.txt loaded as numpy array using Pandas (tab-delimited).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (start_indices, end_indices, counts) for each shutter.

    """
    start_indices = []
    end_indices = []
    counts = []

    prev_time = 0.0
    for number, count in shutter_count:
        if count == 0:
            break

        delay = shutter_times[number, 1]
        duration = shutter_times[number, 2]

        start_time = prev_time + delay
        end_time = start_time + duration
        prev_time = end_time

        start_indices.append(int(np.searchsorted(spectra[:, 0], start_time)))
        end_indices.append(int(np.searchsorted(spectra[:, 0], end_time)))
        counts.append(count)

    start_indices = np.asarray(start_indices, dtype=int)
    end_indices = np.asarray(end_indices, dtype=int)
    counts = np.asarray(counts, dtype=int)

    gap = end_indices[:-1] != start_indices[1:]
    start_indices[1:][gap] -= 1

    return start_indices, end_indices, counts


def normalise_stack_array(
    original_stack: Stack,
    open_beam: Stack,
    window_half: int = 5,
    sum_neighbourhood: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Normalise one experiment stack against an open-beam stack.

    Parameters
    ----------
    original_stack : Stack
        Experiment stack, as read in by the reading step.
    open_beam : Stack
        Open-beam stack (must match the sample stack's shape).
    window_half : int, optional
        Floor of half the width of the averaging kernel (default 5).
    sum_neighbourhood : int, optional
        Size of the frame bins summed to form each open-beam
        reference (default 0, i.e. frame-by-frame).
    scale : float, optional
        Intensity correction factor, i.e. open-beam count / sample count
        (default 1.0).

    Returns
    -------
    np.ndarray
        Normalised float32 stack array (same shape as the input).
    """
    stack = np.asarray(original_stack.data, dtype=np.float32)
    open_beam_stack = np.asarray(open_beam.data, dtype=np.float64)

    if stack.shape != open_beam_stack.shape:
        raise ValueError(
            f"stack shape mismatch: sample {stack.shape} vs open-beam {open_beam_stack.shape}"
        )

    total = stack.shape[0]
    output = np.empty_like(stack, dtype=np.float32)

    for index in range(total):
        start = max(0, index - sum_neighbourhood)
        end = min(total - 1, index + sum_neighbourhood)
        open_beam_sum = (
            open_beam_stack[index]
            if start == end
            else open_beam_stack[start : end + 1].sum(axis=0)
        )
        frame_count = end - start + 1
        output[index] = normalise_frame(
            stack[index], open_beam_sum, window_half, frame_count, scale
        )

    return output


def stack_normalisation(
    stacks: List[Stack],
    open_beam: Stack,
    window_half: int = 5,
    sum_neighbourhood: int = 0,
    scale: float = 1.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> List[Stack]:
    """
    NEAT-normalise every stack against a shared open-beam stack.

    Parameters
    ----------
    stacks : List[Stack]
        Experiment stacks, as read in by the reading step.
    open_beam : Stack
        Open-beam stack (must match each sample stack's shape).
    window_half : int, optional
        Floor of half the width of the averaging kernel (default 5).
    sum_neighbourhood : int, optional
        Size of the frame bins summed to form each open-beam
        reference (default 0, i.e. frame-by-frame).
    scale : float, optional
        Intensity correction factor, i.e. open-beam count / sample count
        (default 1.0).
    progress_callback : Callable[[int, int], None], optional
        Called with (stacks done, total) after each stack, if provided.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        Normalised stacks.
    """
    out = []
    for index, stack in enumerate(stacks, 1):
        normalised = normalise_stack_array(
            stack, open_beam, window_half, sum_neighbourhood, scale
        )
        out.append(Stack.from_array(normalised, stack.headers, stack.stack_meta))
        if progress_callback is not None:
            progress_callback(index, len(stacks))
    return record_derivation(
        out,
        "Normalisation",
        {
            "window_half": window_half,
            "sum_neighbourhood": sum_neighbourhood,
            "scale": scale,
        },
        stacks,
        aux={"open_beam": open_beam},
        mode="map",
    )


def _register_frames_parallel(
    worker: Callable[[np.ndarray], np.ndarray],
    frames: List[np.ndarray],
    max_workers: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Apply registration worker to each frame for parallel processing.
    Falls back to serial execution for a single frame or for exceptions when attempting multiple workers.

    Parameters
    ----------
    worker : Callable[[np.ndarray], np.ndarray]
        Per-frame registration callable.
    frames : List[np.ndarray]
        Frames to register, in order.
    max_workers : int, optional
        Maximum number of worker processes. limited by number of frames and CPU cores.

    Returns
    -------
    List[np.ndarray]
        Registered frames, in the same order as frames.
    """
    n = len(frames)
    if n <= 1:
        return [worker(frame) for frame in frames]
    requested = max_workers or psutil.cpu_count(logical=False) or 1
    workers = max(1, min(requested, n))
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(worker, frames))
        except (BrokenProcessPool, OSError) as exc:
            _log.warning(
                "Parallel registration unavailable (%s); running serially.", exc
            )
    return [worker(frame) for frame in frames]


def _register_frame_safe(
    frame: np.ndarray,
    ref_keypoints: np.ndarray,
    ref_descriptors: np.ndarray,
    feat_keypoints: int,
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Register one frame.
    Returns (frame, error) instead of raising if an exception occurs.

    Returns
    -------
    Tuple[np.ndarray, Optional[str]]
        (registered or original frame, error message or None).
    """
    try:
        registered = register_frame_to_features(
            frame, ref_keypoints, ref_descriptors, feat_keypoints
        )
        return registered, None
    except Exception as exc:
        return np.asarray(frame, dtype=np.float32), str(exc)


def stack_registration(
    stacks: list[Stack],
    reference_img: Stack,
    keypoints: int = 200,
    max_workers: int = max(1, (psutil.cpu_count(logical=False) or 1)),
    **kwargs,
) -> list[Stack]:
    """
    Align stack frames with a reference image via similarity transform.
    Reference image fed in via Stack. Frames registered in parallel.

    Parameters
    ----------
    stacks: list[Stack]
        List of Stacks to align
    reference_img: Stack
        Reference image in stack form.
    keypoints: int, optional
        Number of feature keypoints to use in alignment (default 200).
    max_workers: int, optional
        Max number of worker processes (bound by frame count or cpu core count).
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    list[Stack]
        Processed stacks.

    Raises
    ------
    ValueError
        If a 2D reference frame cannot be derived from reference_img.
    """
    ref = np.asarray(reference_img.data)
    if ref.ndim == 3:
        # Take mean to 2D image
        ref = ref[0] if ref.shape[0] == 1 else np.nanmean(ref, axis=0)
    if ref.ndim != 2:
        raise ValueError(
            f"Reference must reduce to a 2D image, got shape {reference_img.data.shape}."
        )

    # Extract keypoints
    ref_keypoints, ref_descriptors = extract_features(ref, keypoints)
    worker = partial(
        _register_frame_safe,
        ref_keypoints=ref_keypoints,
        ref_descriptors=ref_descriptors,
        feat_keypoints=keypoints,
    )

    # Flatten across all stacks so a single pool does the job.
    frame_lists = [
        [stack.data[i, :, :] for i in range(stack.data.shape[0])] for stack in stacks
    ]
    results = _register_frames_parallel(
        worker, [frame for frames in frame_lists for frame in frames], max_workers
    )
    errors = [err for _, err in results if err is not None]
    if errors and len(errors) == len(results):
        raise ValueError(
            f"No frames could be registered onto the reference "
            f"({len(results)} frame(s) failed). First error: {errors[0]}"
        )
    if errors:
        _log.warning(
            "stack_registration: %d of %d frame(s) could not be registered and "
            "were left unchanged. First error: %s",
            len(errors),
            len(results),
            errors[0],
        )

    out = []
    pos = 0
    for stack, frames in zip(stacks, frame_lists):
        n_slices = len(frames)
        chunk = results[pos : pos + n_slices]
        pos += n_slices
        data = np.stack([arr for arr, _ in chunk])
        skipped = [i for i, (_, err) in enumerate(chunk) if err is not None]
        stack_meta = dict(stack.stack_meta or {})
        if skipped:
            stack_meta["registration_skipped_frames"] = skipped
        out.append(Stack.from_array(data, stack.headers, stack_meta))
    return record_derivation(
        out,
        "Stack Registration",
        {"keypoints": int(keypoints)},
        stacks,
        aux={"reference": reference_img},
        mode="map",
    )


def stack_stitching(
    short: Stack,
    long: Stack,
    short_range: Optional[Sequence[int]] = None,
    long_range: Optional[Sequence[int]] = None,
    delay: float = 0.0,
    collimation_distance: float = 0,
    **kwargs,
) -> List[Stack]:
    """
    Stitch a short- and a long-wavelength stack into one continuous stack.
    Overlapping region is cross-faded.

    Parameters
    ----------
    short : Stack
        The short-wavelength stack.
    long : Stack
        The long-wavelength stack.
    short_range, long_range : sequence of int, optional
        (start, stop) frame ranges to take from each stack; the whole stack by
        default.
    delay : float, optional
        Acquisition delay of the long stack, in the spectra table's time units.
    collimation_distance : float, optional
        Flight-path length used to convert times of flight to wavelengths.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        A single-element list holding the stitched stack.

    Raises
    ------
    ValueError
        If the stacks' frame dimensions differ, or the chosen ranges are empty.
    """
    if short.data.shape[1:] != long.data.shape[1:]:
        raise ValueError(
            f"Stacks have different frame dimensions ({short.data.shape[1:]} vs "
            f"{long.data.shape[1:]}); they must share height and width to stitch."
        )
    s0, s1 = _clamp_frame_range(short_range, short.data.shape[0])
    l0, l1 = _clamp_frame_range(long_range, long.data.shape[0])
    if (s1 - s0) + (l1 - l0) == 0:
        raise ValueError("The selected frame ranges are empty; nothing to stitch.")

    overlap = _stitch_overlap(
        short, long, (s0, s1), (l0, l1), delay, collimation_distance
    )
    # Cross-fade
    crossfade = [
        (1 - (i / overlap)) * short.data[s1 - overlap + i]
        + (i / overlap) * long.data[l0 + i]
        for i in range(overlap)
    ]
    data = np.concatenate(
        [
            short.data[s0 : s1 - overlap],
            np.asarray(crossfade).reshape(overlap, *short.data.shape[1:]),
            long.data[l0 + overlap : l1],
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    info = (
        _stitch_info(short, s0, s1 - overlap, "short")
        + _stitch_info(short, s1 - overlap, s1, "crossfade")
        + _stitch_info(long, l0 + overlap, l1, "long")
    )

    short_times = _kept_spectra_times(short, range(s0, s1))
    long_times = _kept_spectra_times(long, range(l0 + overlap, l1)) + delay
    times = (
        np.concatenate([short_times, long_times])
        if short_times is not None and long_times is not None
        else None
    )
    meta = {
        "spectra_times": (
            times if times is not None and len(times) == len(data) else None
        )
    }

    return record_derivation(
        [Stack(data=data, headers=info, stack_meta=meta, path=None)],
        "Stack Stitching",
        {
            "short_range": [s0, s1],
            "long_range": [l0, l1],
            "delay": float(delay),
            "collimation_distance": float(collimation_distance),
        },
        [short, long],
        mode="reduce",
    )


def _clamp_frame_range(
    frame_range: Optional[Sequence[int]], n_frames: int
) -> Tuple[int, int]:
    """Clamp a (start, stop) pair into [0, n_frames], defaulting to the whole stack."""
    if not frame_range:
        return 0, int(n_frames)
    start, stop = (int(v) for v in frame_range)
    start = max(0, min(start, n_frames))
    stop = max(start, min(stop, n_frames))
    return start, stop


def _stitch_overlap(
    short: Stack,
    long: Stack,
    short_range: Tuple[int, int],
    long_range: Tuple[int, int],
    delay: float,
    collimation_distance: float,
) -> int:
    """
    Number of overlapping frames shared by the two stacks.
    """
    short_wavelengths = short.wavelengths(delay, collimation_distance, False)
    long_wavelengths = long.wavelengths(delay, collimation_distance, True)
    if short_wavelengths is None or long_wavelengths is None:
        return 0

    short_wavelengths = short_wavelengths[slice(*short_range)]
    long_wavelengths = long_wavelengths[slice(*long_range)]
    if not short_wavelengths.size or not long_wavelengths.size:
        return 0
    return int(np.count_nonzero(long_wavelengths < short_wavelengths[-1]))


def _stitch_info(stack: Stack, start: int, stop: int, band: str) -> List[fits.Header]:
    """Per-frame fits.Header for a stitched frame range, annotated with its origin."""
    headers = list(getattr(stack, "headers", []) or [])
    note = f"Stack Stitching: {band} frames [{start}:{stop})"
    out = []
    for i in range(start, stop):
        hdr = headers[i] if 0 <= i < len(headers) else fits.Header()
        hdr.update({"HISTORY": note})
        out.append(hdr)
    return out
