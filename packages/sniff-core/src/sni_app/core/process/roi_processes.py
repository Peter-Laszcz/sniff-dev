"""
ROI analysis and processing functions.

Includes statistics, relative attenuation, attenuation coefficient and cross-sections.

The compute_ functions return arrays with stats.
The stack processes at the bottom of the module return new stacks with lineage.
"""

import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
from molmass import Formula
from numpy.polynomial import polynomial
from scipy import constants
from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
from scipy.ndimage import median_filter as _scipy_median_filter

from sni_app.core.components.stack import (
    Stack,
    record_derivation,
)

BARNS_PER_CM2 = 1e24
"""
Barns per square centimetre, converting between microscopic cross sections
(barns) and macroscopic attenuation coefficients (cm^-1).
"""

JANIS_CATALOGUE: Dict[str, Tuple[polynomial.Polynomial, Tuple[float, float]]] = {
    "C": (
        polynomial.Polynomial([4.7129, 0.0161, 0.0714, -0.0027, 4e-05]),
        (0.286014, 28.60144),
    ),
    "F": (
        polynomial.Polynomial([3.6471, -0.0128, 0.04, -0.0014, 2e-05]),
        (0.286014, 28.60144),
    ),
    "H": (
        polynomial.Polynomial([15.974, 7.8204, 0.6056, -0.028, 0.0004]),
        (0.286014, 28.60144),
    ),
    "Li": (
        polynomial.Polynomial([0.9519, 39.624, 0.0283, -0.0015, 3e-05]),
        (0.29303, 27.88093),
    ),
    "Na": (
        polynomial.Polynomial([3.239, 0.3264, 0.0255, -0.0008, 8e-06]),
        (0.305762, 28.60144),
    ),
    "O": (
        polynomial.Polynomial([3.7936, -0.012, 0.0474, -0.0017, 2e-05]),
        (0.286014, 28.60144),
    ),
    "P": (
        polynomial.Polynomial([4.1162, 0.0679, 0.0295, -0.0009, 1e-05]),
        (0.286014, 28.60144),
    ),
}
"""
Per-element microscopic cross sections (barns) from the JANIS dataset.
Data has been fitted to 4th order polynomial to allow interpolation at arbitrary wavelength.
Original data range is also specified to avoid extrapolation..
"""


def clamp_roi_to_stack(
    roi: tuple[int, int, int, int], stack: Stack
) -> tuple[int, int, int, int]:
    """
    Clamp a ROI to within a stack's frame bounds.

    Parameters
    ----------
    roi : tuple
        ROI in (x,y,w,h) format.
    stack : Stack
        Stack to clamp against.

    Returns
    -------
    tuple[int, int, int, int]
        A clamped ROI (x,y,w,h) with width and height at least 1
        and the rectangle contained in the stack frame.
    """
    _, stack_h, stack_w = stack.data.shape
    x_0, y_0, roi_w, roi_h = [int(v) for v in roi]
    if stack_h <= 0 or stack_w <= 0:
        return (0, 0, 1, 1)  # defaults for malformed arrays
    x_0 = max(0, min(x_0, stack_w - 1))
    y_0 = max(0, min(y_0, stack_h - 1))
    x1 = max(x_0 + 1, min(x_0 + max(1, roi_w), stack_w))
    y1 = max(y_0 + 1, min(y_0 + max(1, roi_h), stack_h))
    return (x_0, y_0, (x1 - x_0), (y1 - y_0))


def roi_profile(roi: Tuple[int, int, int, int], stack: Stack) -> np.ndarray:
    """
    Compute the per-frame mean intensity within an ROI.
    ROI is clamped to the stack before a mean is taken.

    Parameters
    ----------
    roi : Tuple[int, int, int, int]
        Rectangular ROI as (x,y,w,h).
    stack : Stack
        Stack to profile.

    Returns
    -------
    np.ndarray
        float64 array of mean ROI intensity (one value per slice).
    """
    x, y, w, h = clamp_roi_to_stack(roi, stack)
    region = stack.data[:, y : y + h, x : x + w]
    return np.nanmean(region, axis=(1, 2)).astype(np.float64, copy=False)


def compute_roi_stats(
    frame: np.ndarray, roi: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """
    Compute summary stats over a given ROI.

    Non-finite pixels are excluded from all stats (except the invalid_pixels).
    Where there are no valid pixels, all stats return NaN.

    Parameters
    ----------
    frame : np.ndarray
        Image over which to calculate statistics.
    roi : Tuple[int, int, int, int]
        ROI in (x,y,w,h) format.

    Returns
    -------
    Dict[str, float]
        Stats: mean, median, standard deviation, standard error of mean, minimum, maximum,
        number of valid/invalid pixels.
    """
    vals = roi_to_mask(frame, roi)
    finite = np.isfinite(vals)
    valid_vals = vals[finite]
    valid_pixels = int(valid_vals.size)
    total_pixels = int(vals.size)
    invalid_pixels = total_pixels - valid_pixels

    if valid_pixels == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "min": np.nan,
            "max": np.nan,
            "valid_pixels": valid_pixels,
            "valid_n": valid_pixels,
            "invalid_pixels": invalid_pixels,
        }

    std = float(np.nanstd(valid_vals))
    sem = (
        float(std / np.sqrt(valid_pixels))
        if valid_pixels > 1 and np.isfinite(std)
        else np.nan
    )

    return {
        "mean": float(np.nanmean(valid_vals)),
        "median": float(np.nanmedian(valid_vals)),
        "std": std,
        "sem": sem,
        "min": float(np.nanmin(valid_vals)),
        "max": float(np.nanmax(valid_vals)),
        "valid_pixels": valid_pixels,
        "valid_n": valid_pixels,
        "invalid_pixels": invalid_pixels,
    }


def roi_to_mask(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Represents ROI as boolean mask in scope of a dimensionally guiding image.
    Parameters
    ----------
    frame : np.ndarray
        Image which provides shape parameters for mask.
    roi: Tuple[int,int,int,int]
        Form : (x,y,w,h)

    Returns
    -------
    np.ndarray
        Boolean mask representing ROI pixels.
    """
    x, y, w, h = roi
    i_h, i_w = frame.shape
    mask = np.zeros((i_h, i_w), dtype=bool)
    mask[y : y + h, x : x + h] = True
    vals = np.asarray(frame, dtype=np.float32)[mask]
    return vals


def _block_bin_mean(frame: np.ndarray, factor: int) -> np.ndarray:
    """
    Spatially down-bin an image using a square kernel of specified width.
    Edges are cropped to fit bins.

    Parameters
    ----------
    frame : np.ndarray
        Image to bin.
    factor : int
        Kernel width to bin into one pixel.

    Returns
    -------
    np.ndarray
        Binned image.
    """
    arr = np.asarray(frame, dtype=np.float32)
    f = max(1, int(factor))
    if f == 1:
        return arr
    h, w = arr.shape
    h2 = (h // f) * f
    w2 = (w // f) * f
    if h2 <= 0 or w2 <= 0:
        return arr
    arr = arr[:h2, :w2]
    return arr.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3), dtype=np.float32)


def _apply_prefilter(
    frame: np.ndarray,
    enabled: bool,
    mode: str,
    median_size: int = 0,
    gauss_sigma: float = 0.0,
) -> Tuple[np.ndarray, str]:
    """
    Apply a median or Gaussian smoothing filter to an image.

    Parameters
    ----------
    frame : np.ndarray
        Image to filter.
    enabled : bool
        Whether filter is enabled.
    mode : str
        Filter to apply: "Median", "Gaussian" or "None".
    median_size : int
        Median kernel size (forced odd) when mode is set to "Median".
    gauss_sigma : float
        Gaussian standard deviation when mode is set to "Gaussian".

    Returns
    -------
    Tuple[np.ndarray, str]
        Processed image with filter type descriptor.
    """
    arr = np.asarray(frame, dtype=np.float32)
    if (not enabled) or mode == "None" or int(median_size) == 1:
        return arr, "None"
    if mode == "Median":
        size = int(median_size)
        if size % 2 == 0:
            size += 1
        med = (
            _scipy_median_filter(arr, size=size, mode="reflect").astype(
                np.float32, copy=False
            )
            if size <= 1
            else arr
        )
        return med, f"Median(size={size})"
    if mode == "Gaussian":
        gauss = (
            _scipy_gaussian_filter(
                arr, sigma=float(gauss_sigma), mode="reflect"
            ).astype(np.float32, copy=False)
            if gauss_sigma <= 0
            else arr
        )
        return (gauss, f"Gaussian(sigma={float(gauss_sigma):.3g})")
    return arr, "None"


def _frame_means(stack: Stack) -> np.ndarray:
    """
    Reduce a stack to one value per frame by nanmean over each frame.

    Parameters
    ----------
    stack : Stack
        Stack to reduce.

    Returns
    -------
    np.ndarray
        One float64 value per frame; NaN for a frame with no finite pixels.
    """
    data = np.asarray(stack.data, dtype=np.float64)
    with warnings.catch_warnings():
        # An all-NaN frame is a legitimate input and yields NaN, not a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(data, axis=(1, 2))


def _relatt_stats(
    denominator: np.ndarray,
    sw_count: np.ndarray,
    lw_count: np.ndarray,
    rel: np.ndarray,
    eps: float,
    stack_mean: float,
) -> Dict[str, float]:
    """
    Build the diagnostics both relative-attenuation modes report.

    Parameters
    ----------
    denominator : np.ndarray
        The short-band log quantity the division is by.
    sw_count, lw_count : np.ndarray
        Per-pixel counts of finite log values contributing to each wavelength band.
    rel : np.ndarray
        The computed relative attenuation map.
    eps : float
        Minimum absolute denominator for a valid pixel.
    stack_mean : float
        Mean of the source stack, or NaN when no stack is in scope.

    Returns
    -------
    Dict[str, float]
        Band counts, denominator range/percentiles, valid/invalid counts and
        the source stack's mean.
    """

    def _min_max_med(x: np.ndarray) -> Tuple[float, float, float]:
        """(min, max, median) of the finite values of x, or NaNs if empty."""
        f = x[np.isfinite(x)]
        if f.size == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.nanmin(f)), float(np.nanmax(f)), float(np.nanmedian(f)))

    swc_min, swc_max, swc_med = _min_max_med(sw_count)
    lwc_min, lwc_max, lwc_med = _min_max_med(lw_count)
    den = denominator[np.isfinite(denominator)]
    if den.size:
        den_min = float(np.nanmin(den))
        den_max = float(np.nanmax(den))
        den_med = float(np.nanmedian(den))
        den_p01 = float(np.nanpercentile(den, 1))
        den_p99 = float(np.nanpercentile(den, 99))
    else:
        den_min = den_max = den_med = den_p01 = den_p99 = np.nan

    valid_count = int(np.count_nonzero(np.isfinite(rel)))
    return {
        "sw_count_min": swc_min,
        "sw_count_max": swc_max,
        "sw_count_median": swc_med,
        "lw_count_min": lwc_min,
        "lw_count_max": lwc_max,
        "lw_count_median": lwc_med,
        "den_min": den_min,
        "den_max": den_max,
        "den_median": den_med,
        "den_p01": den_p01,
        "den_p99": den_p99,
        "den_abs_lt_eps_count": int(
            np.count_nonzero(
                np.isfinite(denominator) & (np.abs(denominator) < float(eps))
            )
        ),
        "valid_count": valid_count,
        "invalid_count": int(rel.size - valid_count),
        "stack_mean": float(stack_mean),
    }


def _compute_sum_of_logs_relatt_exact(
    stack: Stack,
    sw_range: tuple[int, int],
    lw_range: tuple[int, int],
    eps: float,
    bin_factor: int,
    filter_mode: str,
    filter_enabled: bool,
    median_size: int,
    gauss_sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Compute the exact sum-of-logs relative attenuation map for a stack.

    Relative attenuation = sum(log(long-band))/sum(log(short-band))
    Infeasible pixels marked as invalid.


    Parameters
    ----------
    stack : Stack
        Source stack (transmission data).
    sw_range : tuple[int, int]
        (start, stop) frame indices of the short-wavelength band.
    lw_range : tuple[int, int]
        (start, stop) frame indices of the long-wavelength band.
    eps : float
        Minimum absolute short-band log-sum (denominator) for a valid pixel.
    bin_factor : int
        Spatial block-binning factor applied to the summed maps.
    filter_mode : str
        Pre-filter mode passed to :func:`_apply_prefilter`.
    filter_enabled : bool
        Whether pre-filtering is applied.
    median_size : int
        Median kernel size for the pre-filter.
    gauss_sigma : float
        Gaussian sigma for the pre-filter.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        The relative-attenuation map and the shared diagnostics dict built by _relatt_stats.
    """
    arr = stack.data
    sw0, sw1 = sw_range
    lw0, lw1 = lw_range
    sw_band = np.asarray(arr[sw0:sw1, :, :], dtype=np.float32)
    lw_band = np.asarray(arr[lw0:lw1, :, :], dtype=np.float32)

    sw_log = np.full(sw_band.shape, np.nan, dtype=np.float32)
    lw_log = np.full(lw_band.shape, np.nan, dtype=np.float32)
    sw_valid = np.isfinite(sw_band) & (sw_band > 0)
    lw_valid = np.isfinite(lw_band) & (lw_band > 0)
    sw_log[sw_valid] = np.log(sw_band[sw_valid]).astype(np.float32, copy=False)
    lw_log[lw_valid] = np.log(lw_band[lw_valid]).astype(np.float32, copy=False)

    sw_log_sum = np.nansum(sw_log, axis=0).astype(np.float32, copy=False)
    lw_log_sum = np.nansum(lw_log, axis=0).astype(np.float32, copy=False)

    sw_count = np.count_nonzero(np.isfinite(sw_log), axis=0)
    lw_count = np.count_nonzero(np.isfinite(lw_log), axis=0)
    sw_log_sum = _block_bin_mean(sw_log_sum, factor=bin_factor)
    lw_log_sum = _block_bin_mean(lw_log_sum, factor=bin_factor)
    sw_count = _block_bin_mean(sw_count.astype(np.float32), factor=bin_factor)
    lw_count = _block_bin_mean(lw_count.astype(np.float32), factor=bin_factor)

    sw_log_sum, _ = _apply_prefilter(
        sw_log_sum, filter_enabled, filter_mode, median_size, gauss_sigma
    )
    lw_log_sum, _ = _apply_prefilter(
        lw_log_sum, filter_enabled, filter_mode, median_size, gauss_sigma
    )
    sw_count, _ = _apply_prefilter(
        sw_count, filter_enabled, filter_mode, median_size, gauss_sigma
    )
    lw_count, _ = _apply_prefilter(
        lw_count, filter_enabled, filter_mode, median_size, gauss_sigma
    )

    rel = np.full(sw_log_sum.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(sw_log_sum)
        & np.isfinite(lw_log_sum)
        & (np.abs(sw_log_sum) >= float(eps))
        & (sw_count > 0)
        & (lw_count > 0)
    )
    rel[valid] = (lw_log_sum[valid] / sw_log_sum[valid]).astype(np.float32, copy=False)

    stats = _relatt_stats(
        sw_log_sum, sw_count, lw_count, rel, eps, float(np.nanmean(_frame_means(stack)))
    )
    return rel, stats


def _compute_relative_attenuation(
    sw_img: np.ndarray,
    lw_img: np.ndarray,
    eps: float,
    stack_mean: float = np.nan,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Relative attenuation from two pre-summed short-/long-wavelength images.

    Computes log(lw) / log(sw) per pixel where both inputs are finite and
    positive and the short-band log magnitude is at least eps; other pixels
    are marked invalid (NaN).

    Parameters
    ----------
    sw_img : np.ndarray
        Short-wavelength (denominator) image.
    lw_img : np.ndarray
        Long-wavelength (numerator) image.
    eps : float
        Minimum absolute short-band log value for a valid pixel.
    stack_mean : float, optional
        Mean of the stack the two images came from, reported in the stats.
        Defaults to NaN, since two bare images do not carry one; the
        stack-level callers supply it.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        The relative-attenuation map and the shared diagnostics dict built by
        _relatt_stats, the same schema the sum-of-logs mode reports.
        Each pixel here is backed by a single band image, so the band counts
        are 1 where that image contributed and 0 where it did not.
    """
    sw = np.asarray(sw_img, dtype=np.float32)
    lw = np.asarray(lw_img, dtype=np.float32)
    rel = np.full(sw.shape, np.nan, dtype=np.float32)

    finite = np.isfinite(sw) & np.isfinite(lw)
    pos = finite & (sw > 0) & (lw > 0)

    sw_log = np.full(sw.shape, np.nan, dtype=np.float32)
    lw_log = np.full(sw.shape, np.nan, dtype=np.float32)
    sw_log[pos] = np.log(sw[pos]).astype(np.float32, copy=False)
    lw_log[pos] = np.log(lw[pos]).astype(np.float32, copy=False)

    valid = (
        pos & np.isfinite(sw_log) & np.isfinite(lw_log) & (np.abs(sw_log) >= float(eps))
    )
    rel[valid] = (lw_log[valid] / sw_log[valid]).astype(np.float32, copy=False)

    stats = _relatt_stats(
        sw_log,
        np.isfinite(sw_log).astype(np.float32),
        np.isfinite(lw_log).astype(np.float32),
        rel,
        eps,
        stack_mean,
    )
    return rel, stats


def _compute_relative_attenuation_from_stacks(
    sw_stack: Stack,
    lw_stack: Stack,
    eps: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Relative attenuation directly from two stacks.

    Each stack is reduced to a 2-D image by nanmean over its frames, then
    passed to _compute_relative_attenuation.

    Parameters
    ----------
    sw_stack : Stack
        Short-wavelength (denominator) stack.
    lw_stack : Stack
        Long-wavelength (numerator) stack.
    eps : float
        Minimum absolute short-band log value for a valid pixel.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        As for _compute_relative_attenuation.
    """
    sw_img = np.nanmean(np.asarray(sw_stack.data, dtype=np.float32), axis=0)
    lw_img = np.nanmean(np.asarray(lw_stack.data, dtype=np.float32), axis=0)
    both = np.concatenate([_frame_means(sw_stack), _frame_means(lw_stack)])
    return _compute_relative_attenuation(sw_img, lw_img, eps, float(np.nanmean(both)))


def _compute_relative_attenuation_from_bands(
    stack: Stack,
    sw_range: Tuple[int, int],
    lw_range: Tuple[int, int],
    eps: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Relative attenuation over a single given stack.

    Parameters
    ----------
    stack : Stack
        Source stack.
    sw_range, lw_range : tuple[int, int]
        frame ranges of the short- and long-wavelength bands.
    eps : float
        Minimum absolute short-band log value for a valid pixel.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        From _compute_relative_attenuation.
    """
    data = np.asarray(stack.data, dtype=np.float32)
    sw0, sw1 = sw_range
    lw0, lw1 = lw_range
    sw_img = np.nanmean(data[sw0:sw1, :, :], axis=0)
    lw_img = np.nanmean(data[lw0:lw1, :, :], axis=0)
    return _compute_relative_attenuation(
        sw_img, lw_img, eps, float(np.nanmean(_frame_means(stack)))
    )


def _compute_atten_coefficient_from_stacks(
    stack: Stack, empty_holder_stack: Stack, d_cm: float
) -> np.ndarray:
    """
    Attenuation coefficient computed against an empty sample holder stack.

    Both sides are reduced to one value per frame by nanmean before the
    division, so the result is a spectrum rather than an image. Result is
    passed to _compute_atten_coeff_stack.

    Parameters
    ----------
    stack : Stack
        Transmission stack to convert to sigma.
    empty_holder_stack : Stack
        Empty sample holder stack, whose per-frame means provide the
        normalisation values; must have the same frame count as *stack*.
    d_cm : float
        Sample thickness in centimetres.

    Returns
    -------
    np.ndarray
        The attenuation coefficient plot, shaped (z, 1, 1).

    Raises
    ------
    ValueError
        If the frame counts differ or d_cm <= 0.
    """
    return _compute_atten_coeff_stack(stack, _frame_means(empty_holder_stack), d_cm)


def _compute_atten_coeff_stack(
    stack: Stack, empty_holder_values: np.ndarray, d_cm: float
) -> np.ndarray:
    """
    Compute a macroscopic cross-section (sigma) spectrum from transmission data.

    The stack is reduced to one value per frame by nanmean, matching the
    per-frame empty sample holder values it is divided by, and
    sigma = -log(T / empty_holder_values[i]) / d_cm is evaluated once per
    frame. Frames whose mean or holder value is non-positive or non-finite
    yield NaN.

    Parameters
    ----------
    stack : Stack
        input stack.
    empty_holder_values : np.ndarray
        Per-frame empty sample holder normalisation values; length must equal
        the stack's frame count (z).
    d_cm : float
        Sample thickness in centimetres; must be positive.

    Returns
    -------
    np.ndarray
        An attenuation coefficient plot shaped (z, 1, 1), invalid frames NaN.

    Raises
    ------
    ValueError
        If len(empty_holder_values) != z or d_cm <= 0.
    """
    z = int(np.asarray(stack.data).shape[0])
    mt = np.asarray(empty_holder_values, dtype=np.float64)
    if len(mt) != z:
        raise ValueError(
            f"Empty sample holder length {len(mt)} does not match stack Z length {z}"
        )
    if d_cm <= 0:
        raise ValueError("d must be > 0")

    means = _frame_means(stack)
    out = np.full((z, 1, 1), np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = means / mt
    valid = (
        np.isfinite(means)
        & np.isfinite(mt)
        & (means > 0)
        & (mt > 0)
        & np.isfinite(t)
        & (t > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid, 0, 0] = (-np.log(t[valid]) / d_cm).astype(np.float32, copy=False)
    return out


def stack_wavelengths(stack: Stack) -> np.ndarray:
    """
    Return a stack's per-frame neutron wavelengths, in angstroms.

    Reads the wavelengths recorded on the stack, falling back to deriving them
    from its times of flight (see Stack.wavelengths).

    Parameters
    ----------
    stack : Stack
        Stack to read.

    Returns
    -------
    np.ndarray
        One float64 wavelength per frame.

    Raises
    ------
    ValueError
        If the stack carries neither wavelengths nor the frame times they are
        derived from, or holds fewer wavelengths than frames.
    """
    wavelengths = stack.stack_meta.get("wavelengths")
    if wavelengths is None:
        wavelengths = stack.wavelengths()
    if wavelengths is None:
        raise ValueError(
            "Stack carries no per-frame wavelengths, and none can be derived "
            "from it: load it from a folder holding the experiment spectra."
        )
    wavelengths = np.asarray(wavelengths, dtype=np.float64).ravel()
    n_frames = int(np.asarray(stack.data).shape[0])
    if wavelengths.size < n_frames:
        raise ValueError(
            f"Stack has {n_frames} frame(s) but only {wavelengths.size} "
            "wavelength(s)."
        )
    return wavelengths[:n_frames]


def _element_number_densities(
    compounds: Sequence[str],
    densities: Sequence[float],
    ratio: Sequence[float] = (1.0,),
    by_volume: bool = False,
) -> Dict[str, float]:
    """
    Per-element number densities of a compound mixture, in atoms per cm^3.

    Compound must be composed of elements found in the JANIS catalogue dataset in SNIFF.

    Parameters
    ----------
    compounds : sequence of str
        Chemical formulae of the mixture's compounds, e.g. ["C3H4O3"].
    densities : sequence of float
        Density of each compound, in g/cm^3.
    ratio : sequence of float
        Mixing ratio of the compounds, normalised by its own sum.
    by_volume : bool
        Whether the ratio is by volume rather than by mole.

    Returns
    -------
    Dict[str, float]
        Number density per element symbol, in atoms per cm^3.

    Raises
    ------
    ValueError
        If the three sequences differ in length, if a formula cannot be read,
        or if the ratio sums to zero.
    """
    if not compounds:
        raise ValueError("Give at least one compound.")
    if not (len(compounds) == len(densities) == len(ratio)):
        raise ValueError(
            f"compounds ({len(compounds)}), densities ({len(densities)}) and "
            f"ratio ({len(ratio)}) must be the same length."
        )
    ratio_total = float(sum(ratio))
    if ratio_total == 0:
        raise ValueError("The mixing ratio must not sum to zero.")

    try:
        formulae = [Formula(str(compound)) for compound in compounds]
    except Exception as exc:
        raise ValueError(f"Cannot read a compound formula: {exc}")

    moles = [
        (
            float(densities[i]) * formulae[i].mass
            if by_volume
            else float(densities[i]) / formulae[i].mass
        )
        * float(ratio[i])
        / ratio_total
        for i in range(len(formulae))
    ]

    number_densities: Dict[str, float] = {}
    for formula, compound_moles in zip(formulae, moles):
        composition = formula.composition().dataframe()
        for element in JANIS_CATALOGUE:
            if element not in composition.index:
                continue
            count = float(composition.loc[element, "Count"])
            n_j = count * compound_moles * constants.Avogadro
            number_densities[element] = number_densities.get(element, 0.0) + n_j
    return number_densities


def _compute_total_micro_cross_section(
    stack: Stack, molar_mass: float, density: float, d_cm: float
) -> np.ndarray:
    """
    Total microscopic cross-section plot from a transmission stack.

    The stack is reduced to one transmission value per frame by nanmean and
    sigma = -ln(T) * M / (rho * d * N_A) is evaluated once per frame, reported
    in barns, where T is transmission, rho is density, d is sample thickness, and N_A is Avogadro's constant
    Frames whose transmission is non-positive or non-finite yield
    NaN.

    Parameters
    ----------
    stack : Stack
        Transmission stack (already normalised against an open beam).
    molar_mass : float
        Sample molar mass, in g/mol.
    density : float
        Effective density of the sample, in g/cm^3.
    d_cm : float
        Sample thickness in centimetres.

    Returns
    -------
    np.ndarray
        A cross-section spectrum in barns, shaped (z, 1, 1).

    Raises
    ------
    ValueError
        If any of molar_mass, density or d_cm is not positive.
    """
    if molar_mass <= 0:
        raise ValueError("Sample molar mass must be > 0")
    if density <= 0:
        raise ValueError("Effective density must be > 0")
    if d_cm <= 0:
        raise ValueError("d must be > 0")

    transmission = _frame_means(stack)
    out = np.full((transmission.size, 1, 1), np.nan, dtype=np.float32)
    valid = np.isfinite(transmission) & (transmission > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid, 0, 0] = (
            -np.log(transmission[valid])
            * float(molar_mass)
            * BARNS_PER_CM2
            / (float(density) * float(d_cm) * constants.Avogadro)
        ).astype(np.float32, copy=False)
    return out


def _compute_hydrogen_cross_section(
    atten_coeff: np.ndarray,
    wavelengths: np.ndarray,
    number_densities: Dict[str, float],
) -> np.ndarray:
    """
    Hydrogen cross section plot.


    Parameters
    ----------
    atten_coeff : np.ndarray
        Measured attenuation coefficient, one value per frame, in cm^-1.
    wavelengths : np.ndarray
        Wavelength of each frame, in angstroms.
    number_densities : Dict[str, float]
        Number density per element symbol, in atoms per cm^3, as returned by _element_number_densities.

    Returns
    -------
    np.ndarray
        The hydrogen cross-section in barns, shaped (z, 1, 1).

    Raises
    ------
    ValueError
        If the mixture holds no hydrogen, or the wavelength axis is shorter
        than the coefficient spectrum.
    """
    coeff = np.asarray(atten_coeff, dtype=np.float64).ravel()
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    if wl.size < coeff.size:
        raise ValueError(
            f"{coeff.size} coefficient value(s) but only {wl.size} wavelength(s)."
        )
    wl = wl[: coeff.size]

    hydrogen_n_j = float(number_densities.get("H", 0.0))
    if hydrogen_n_j <= 0:
        raise ValueError("The compound mixture holds no hydrogen to solve for.")

    others = [
        JANIS_CATALOGUE[element][0](wl) * n_j
        for element, n_j in number_densities.items()
        if element != "H"
    ]
    catalogued = (
        np.sum(others, axis=0) if others else np.zeros(coeff.size, dtype=np.float64)
    )
    hydrogen = (coeff * BARNS_PER_CM2 - catalogued) / hydrogen_n_j
    return hydrogen.astype(np.float32, copy=False).reshape(-1, 1, 1)


#############################
# STACK-LEVEL ROI PROCESSES #
#############################


def _result_stack(array: np.ndarray, source: Stack, stats: Dict = None) -> Stack:
    array = np.asarray(array)
    frames = 1 if array.ndim == 2 else int(array.shape[0])
    headers = source.headers if len(source.headers) == frames else None
    stack = Stack.from_array(array, headers, source.stack_meta)
    stack.record_analysis_results(stats or {})
    return stack


def roi_to_stack(
    stacks: Sequence[Stack], roi: Tuple[int, int, int, int], **kwargs
) -> List[Stack]:
    """
    Crop given stacks to an ROI.

    The ROI is clamped to each stack independently, so stacks of differing
    dimensions can be cropped in one call.

    Parameters
    ----------
    stacks : sequence of Stack
        Stacks to crop.
    roi : tuple[int, int, int, int]
        ROI in (x, y, w, h) format.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        The cropped stacks, one per input.
    """
    out = []
    for stack in stacks:
        x, y, w, h = clamp_roi_to_stack(tuple(int(v) for v in roi), stack)
        data = np.asarray(stack.data)[:, y : y + h, x : x + w]
        meta = dict(stack.stack_meta)
        meta["spectra_times"] = stack.times_of_flight()
        meta["roi_crop_xywh"] = [int(x), int(y), int(w), int(h)]
        out.append(Stack.from_array(data, stack.headers, meta))
    return record_derivation(
        out,
        "ROI to Stack",
        {"roi_xywh": [int(v) for v in roi]},
        stacks,
        mode="map",
    )


def relative_attenuation(
    stack: Stack,
    sw_range: Tuple[int, int],
    lw_range: Tuple[int, int],
    eps: float = 1e-6,
    **kwargs,
) -> List[Stack]:
    """
    Relative attenuation of a stack's short- and long-wavelength bands.
    Stack-level version of _compute_relative_attenuation_from_bands.

    Parameters
    ----------
    stack : Stack
        Source stack.
    sw_range, lw_range : tuple[int, int]
        Frame ranges of the short- and long-wavelength bands.
    eps : float, optional
        Minimum absolute short-band log value for a valid pixel.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        A single-element list holding the relative-attenuation map, with the
        computation's diagnostics attached as its analysis results.
    """
    array, stats = _compute_relative_attenuation_from_bands(
        stack, tuple(sw_range), tuple(lw_range), float(eps)
    )
    return record_derivation(
        [_result_stack(array, stack, stats)],
        "Relative Attenuation (images)",
        {
            "sw_range": [int(v) for v in sw_range],
            "lw_range": [int(v) for v in lw_range],
            "eps": float(eps),
        },
        [stack],
        mode="reduce",
    )


def sum_of_logs_relative_attenuation(
    stack: Stack,
    sw_range: Tuple[int, int],
    lw_range: Tuple[int, int],
    eps: float = 1e-6,
    bin_factor: int = 1,
    filter_mode: str = "None",
    filter_enabled: bool = False,
    median_size: int = 3,
    gauss_sigma: float = 1.0,
    **kwargs,
) -> List[Stack]:
    """
    Sum-of-logs relative attenuation of a stack's two wavelength bands.
    Stack level version of _compute_sum_of_logs_relatt_exact.

    Returns
    -------
    List[Stack]
        A single-element list holding the relative-attenuation map, with the
        computation's diagnostics attached as its analysis results.
    """
    array, stats = _compute_sum_of_logs_relatt_exact(
        stack,
        tuple(sw_range),
        tuple(lw_range),
        float(eps),
        int(bin_factor),
        filter_mode,
        bool(filter_enabled),
        int(median_size),
        float(gauss_sigma),
    )
    return record_derivation(
        [_result_stack(array, stack, stats)],
        "Rel. Attenuation (sum-of-logs)",
        {
            "sw_range": [int(v) for v in sw_range],
            "lw_range": [int(v) for v in lw_range],
            "eps": float(eps),
            "bin_factor": int(bin_factor),
            "filter_mode": filter_mode,
            "filter_enabled": bool(filter_enabled),
            "median_size": int(median_size),
            "gauss_sigma": float(gauss_sigma),
        },
        [stack],
        mode="reduce",
    )


def atten_coefficient(
    stack: Stack, empty_holder_stack: Stack, d_cm: float = 1.0, **kwargs
) -> List[Stack]:
    """
    Macroscopic cross-section spectrum from transmission data and an empty
    sample holder stack.
    Stack version of _compute_atten_coefficient_from_stacks.

    Both the stack and the empty sample holder are reduced to one value per
    frame by nanmean, so the output holds a single coefficient per frame rather
    than an image: a stack of shape (z, 1, 1), in inverse centimetres.

    Parameters
    ----------
    stack : Stack
        Transmission stack to convert to sigma.
    empty_holder_stack : Stack
        Empty sample holder stack, whose per-frame means provide the
        normalisation values.
    d_cm : float, optional
        Sample thickness in centimetres.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        A single-element list holding the coefficient spectrum, shaped
        (z, 1, 1).
    """
    array = _compute_atten_coefficient_from_stacks(
        stack, empty_holder_stack, float(d_cm)
    )
    return record_derivation(
        [_result_stack(array, stack)],
        "Attenuation Coefficient",
        {"d_cm": float(d_cm)},
        [stack, empty_holder_stack],
        mode="reduce",
    )


def t_cross_section(
    stack: Stack,
    molar_mass: float,
    density: float,
    d_cm: float = 1.0,
    **kwargs,
) -> List[Stack]:
    """
    Total microscopic cross-section spectrum of a sample from its transmission.
    Stack version of _compute_total_micro_cross_section.

    Parameters
    ----------
    stack : Stack
        Transmission stack (already normalised against an open beam).
    molar_mass : float
        Sample molar mass, in g/mol.
    density : float
        Effective density of the sample, in g/cm^3.
    d_cm : float, optional
        Sample thickness in centimetres.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        A single-element list holding the cross-section spectrum, shaped
        (z, 1, 1).
    """
    array = _compute_total_micro_cross_section(
        stack, float(molar_mass), float(density), float(d_cm)
    )
    return record_derivation(
        [_result_stack(array, stack)],
        "Total Microscopic Cross Section",
        {
            "molar_mass": float(molar_mass),
            "density": float(density),
            "d_cm": float(d_cm),
        },
        [stack],
        mode="reduce",
    )


def h_cross_section(
    stack: Stack,
    empty_holder_stack: Stack,
    compounds: Sequence[str],
    densities: Sequence[float],
    d_cm: float = 1.0,
    ratio: Sequence[float] = (1.0,),
    by_volume: bool = False,
    **kwargs,
) -> List[Stack]:
    """
    Hydrogen cross-section spectrum of a compound mixture.


    Parameters
    ----------
    stack : Stack
        Transmission stack of the sample.
    empty_holder_stack : Stack
        Empty sample holder stack, whose per-frame means provide the
        normalisation values.
    compounds : sequence of str
        Chemical formulae of the mixture's compounds, e.g. ["C3H4O3"].
    densities : sequence of float
        Density of each compound, in g/cm^3.
    d_cm : float, optional
        Sample thickness in centimetres.
    ratio : sequence of float, optional
        Mixing ratio of the compounds, normalised by its own sum.
    by_volume : bool, optional
        Whether the ratio is by volume rather than by mole.
    **kwargs
        Ignored extra parameters.

    Returns
    -------
    List[Stack]
        A single-element list holding the hydrogen cross-section spectrum,
        shaped (z, 1, 1).

    Raises
    ------
    ValueError
        If the mixture is not describable (see
        _element_number_densities), holds no hydrogen, or the sample
        carries no per-frame wavelengths.
    """
    compounds = [str(c) for c in compounds]
    densities = [float(v) for v in densities]
    ratio = [float(v) for v in ratio]

    number_densities = _element_number_densities(
        compounds, densities, ratio, bool(by_volume)
    )
    coefficient = atten_coefficient(stack, empty_holder_stack, float(d_cm))[0]
    array = _compute_hydrogen_cross_section(
        coefficient.data, stack_wavelengths(coefficient), number_densities
    )
    return record_derivation(
        [_result_stack(array, coefficient)],
        "Hydrogen Cross Section",
        {
            "compounds": compounds,
            "densities": densities,
            "d_cm": float(d_cm),
            "ratio": ratio,
            "by_volume": bool(by_volume),
        },
        [stack, empty_holder_stack],
        mode="reduce",
    )
