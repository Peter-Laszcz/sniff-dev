"""
Processes on 2D images in array format.
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from skimage import transform
from skimage.feature import ORB, match_descriptors
from skimage.measure import label, ransac

###########################
# REGISTRATION PARAMETERS #
###########################

REGISTRATION_MIN_SAMPLES = 2
"""Minimum correspondences RANSAC fits a similarity transform from."""

MIN_MATCH_COUNT = 6
"""Fewest feature correspondences accepted, so noise is not fitted."""

MIN_INLIER_COUNT = 4
"""Minimal RANSAC inliers accepted."""

MAX_SCALE_DEVIATION = 0.05
"""Maximum transformation scale deviation from 1."""

MAX_ROTATION_DEG = 5.0
"""Maximum transformation rotation, in degrees."""

MAX_TRANSLATION_FRACTION = 0.05
"""Largest transformation shift as a fraction of the shortest frame side."""


def _squeeze_to_2d(frame: np.ndarray) -> np.ndarray:
    """
    Squeeze array without dropping to scalar (1D).

    Parameters
    ----------
    frame : np.ndarray
        Image array.

    Returns
    -------
    np.ndarray
        Squeezed array.
    """
    frame = np.asarray(frame)
    for axis in reversed(range(frame.ndim)):
        if frame.ndim <= 2:
            break
        if frame.shape[axis] == 1:
            frame = np.squeeze(frame, axis=axis)
    return frame


def _frame_to_2d_float32(frame: np.ndarray) -> np.ndarray:
    """
    Cast an image array to a 2D float32 array. Dimensionality of colour images is reduced by averaging the channels.

    Parameters
    ----------
    frame : np.ndarray
        Image array.

    Returns
    -------
    np.ndarray
        2D float32 array.

    Raises
    ------
    ValueError
        If array can't be cast to greyscale 2d.
    """
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame.astype(np.float32, copy=False)

    # convert multichannel to grayscale via channel mean.
    if frame.ndim == 3:
        if frame.shape[-1] >= 3:
            return frame[..., :3].mean(axis=-1, dtype=np.float32)
        if frame.shape[0] >= 3:
            return frame[:3].mean(axis=0, dtype=np.float32)

    raise ValueError(f"Unsupported frame shape: {frame.shape}.")


def _windowed_mean(frame: np.ndarray, window_half: int) -> np.ndarray:
    """
    Arithmetic frame mean using sliding window.

    Parameters
    ----------
    frame : ndarray
        Sample frame from experiment.
    window_half : int
        Half the width of the sliding window (floored).

    Returns
    -------
    ndarray
        Array of means.
    """

    height, width = frame.shape
    side = 2 * window_half + 1
    # Integral image of zero-padded frame
    integral = np.pad(
        np.pad(frame, window_half).cumsum(0).cumsum(1), ((1, 0), (1, 0)), "constant"
    )
    window_sum = (
        integral[side:, side:]
        - integral[:-side, side:]
        - integral[side:, :-side]
        + integral[:-side, :-side]
    )

    rows, cols = np.arange(height), np.arange(width)
    row_counts = np.clip(rows + window_half + 1, 0, height) - np.clip(
        rows - window_half, 0, height
    )
    col_counts = np.clip(cols + window_half + 1, 0, width) - np.clip(
        cols - window_half, 0, width
    )

    return window_sum / np.outer(row_counts, col_counts)


def _normalise_frame(
    frame: np.ndarray,
    beam_frame_sum: np.ndarray,
    window_half: int,
    ob_frame_count: int = 1,
    intensity_correction_scale: float = 1.0,
) -> np.ndarray:
    """
    Normalise a sample frame with a summed open-beam frame.

    Parameters
    ----------
    frame : ndarray
        Input frame.
    beam_frame_sum : ndarray
        Sum of open-beam frames.
    window_half : int
        Half the width of the averaging window (floored).
    ob_frame_count : int
        Number of open-beam frames used to produce beam_frame_sum.
    intensity_correction_scale : float
        Intensity correction factor (open-beam count / sample count).

    Returns
    -------
    ndarray
        Normalised float32 frame.
    """
    if frame.shape != beam_frame_sum.shape:
        raise ValueError(
            f"Shape mismatch: sample {frame.shape} does not match open-beam {beam_frame_sum.shape}."
        )

    height, width = frame.shape
    side = 2 * window_half + 1
    if height < side or width < side:
        raise ValueError(f"Input frame too small for {side}x{side} window")

    thresh = np.float32(1e-7)
    smoothed = _windowed_mean(beam_frame_sum, window_half).astype(np.float32)
    smoothed = np.where(smoothed > 0, smoothed, thresh)

    normalised = (ob_frame_count * frame / smoothed) * np.float32(
        intensity_correction_scale
    )
    return np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


MIN_BLACK_BODIES = 3
"""
Fewest black bodies a background can be fitted through.
"""


class _BlackBodyFit:
    """
    Fits the spatial background of a frame from a black-body mask.

    Parameters
    ----------
    mask : np.ndarray
        2D mask.

    Raises
    ------
    ValueError
        If the mask is not 2D or has too few black bodies.
    """

    def __init__(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(
                f"Black-body mask must be a single image, got {mask.shape}."
            )

        labelled = label(mask > 0)
        count = int(labelled.max())
        if count < MIN_BLACK_BODIES:
            raise ValueError(
                f"Black-body mask marks {count} black "
                f"{'body' if count == 1 else 'bodies'}; fitting a background "
                f"needs at least {MIN_BLACK_BODIES}."
            )

        centroids, regions = [], []
        for index in range(1, count + 1):
            region = labelled == index
            y, x = np.mean(np.argwhere(region), axis=0)
            centroids.append((x, y))
            regions.append(region)

        self.shape = mask.shape
        self.centroids = np.asarray(centroids, dtype=float)
        self.regions = regions
        grid_y, grid_x = np.indices(mask.shape)
        self._query_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def __len__(self) -> int:
        """Number of black bodies the fit is built on."""
        return len(self.regions)

    def background(self, frame: np.ndarray) -> np.ndarray:
        """
        Fit the background across one frame.
        Non-finite pixels are left out of the black-body sample.

        Parameters
        ----------
        frame : np.ndarray
            Image to read the black-body intensities from. Must match the
            mask's shape.

        Returns
        -------
        np.ndarray
            The fitted background, the same shape as frame.

        Raises
        ------
        ValueError
            If frame does not match the mask's shape.
        """
        if frame.shape != self.shape:
            raise ValueError(
                f"Black-body mask {self.shape} does not match frame shape "
                f"{frame.shape}."
            )
        values = np.array([np.nanmean(frame[region]) for region in self.regions])
        interpolator = RBFInterpolator(
            self.centroids, values, kernel="thin_plate_spline", degree=1
        )
        background = interpolator(self._query_points)
        return background.reshape(self.shape).astype(np.float32)


def _robust_percentile_limits(
    frame: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0
) -> Tuple[float, float]:
    """
    Estimate robust lower/upper array value limits from percentiles.
    Large inputs subsampled, non-finite values are ignored.

    Parameters
    ----------
    frame : np.ndarray
        Frame to derive bounds from.
    p_lo : float, optional
        Lower percentile (default 1.0).
    p_hi : float, optional
        Upper percentile (default 99.0).

    Returns
    -------
    Tuple[float, float]
        (lo, hi) display limits.
    """
    arr = np.asarray(frame, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 1.0
    flat = arr.reshape(-1)
    if flat.size > 200_000:
        step = max(1, flat.size // 200_000)
        flat = flat[::step]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(flat, p_lo))
    hi = float(np.percentile(flat, p_hi))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo + 1.0
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _extract_features(
    image: np.ndarray, n_keypoints: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect ORB keypoints and descriptors for an image.

    Parameters
    ----------
    image : np.ndarray
        2D image to extract features from.
    n_keypoints : int, optional
        Number of feature keypoints to extract (default 200).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (keypoints, descriptors).
    """
    orb = ORB(n_keypoints=n_keypoints)
    orb.detect_and_extract(np.asarray(image))
    return orb.keypoints, orb.descriptors


def _reject_implausible_transform(
    model: transform.SimilarityTransform, frame_shape: Tuple[int, ...]
) -> None:
    """
    Raise if a fitted transform is physically infeasible according to predefined bounds.

    Parameters
    ----------
    model : transform.SimilarityTransform
        RANSAC-fitted transform.
    frame_shape : Tuple[int, ...]
        Shape of the frame being registered; its shorter side scales the
        translation bound.

    Raises
    ------
    ValueError
        If the transform's scale, rotation, or translation exceeds the bounds.
    """
    scale = float(getattr(model, "scale", 1.0))
    rotation_deg = abs(np.rad2deg(float(model.rotation)))
    shift = float(np.hypot(*model.translation))
    max_shift = MAX_TRANSLATION_FRACTION * min(frame_shape[:2])
    if (
        abs(scale - 1.0) > MAX_SCALE_DEVIATION
        or rotation_deg > MAX_ROTATION_DEG
        or shift > max_shift
    ):
        raise ValueError(
            f"Implausible registration transform for sample "
            f"(scale={scale:.3f}, rotation={rotation_deg:.1f}deg, shift={shift:.1f}px "
            f"> {max_shift:.1f}px cap); frame left unregistered."
        )


def _register_frame_to_features(
    frame: np.ndarray,
    ref_keypoints: np.ndarray,
    ref_descriptors: np.ndarray,
    feat_keypoints: int = 200,
) -> np.ndarray:
    """
    Warp a single frame onto a reference described by precomputed ORB features.
    This function is executed in parallel.

    Parameters
    ----------
    frame : np.ndarray
        Frame to be aligned.
    ref_keypoints : np.ndarray
        Reference keypoints, as returned by extract_orb_features.
    ref_descriptors : np.ndarray
        Reference descriptors, as returned by extract_orb_features.
    feat_keypoints : int, optional
        Number of feature keypoints extracted from the frame (default 200).

    Returns
    -------
    np.ndarray
        Aligned frame.

    Raises
    ------
    ValueError
        If transform is unreliable (too few feature matches or inliers) or physically implausible.
    """
    frame_keypoints, frame_descriptors = _extract_features(frame, feat_keypoints)

    # Match features
    matches = match_descriptors(ref_descriptors, frame_descriptors, cross_check=True)

    if matches.shape[0] < MIN_MATCH_COUNT:
        raise ValueError(
            f"Only {matches.shape[0]} correspondences between frame and "
            f"reference, needs at least {MIN_MATCH_COUNT} for a reliable "
            f"registration."
        )

    src = frame_keypoints[matches[:, 1]][:, ::-1]  # (y, x) -> (x, y)
    dst = ref_keypoints[matches[:, 0]][:, ::-1]
    model_robust, inliers = ransac(
        (src, dst),
        transform.SimilarityTransform,
        min_samples=REGISTRATION_MIN_SAMPLES,
        residual_threshold=2,
        max_trials=1000,
    )
    if model_robust is None:  # model failed
        raise ValueError(
            "Could not estimate a registration transform "
        )

    n_inliers = int(np.count_nonzero(inliers)) if inliers is not None else 0
    if n_inliers < MIN_INLIER_COUNT:
        raise ValueError(
            f"Registration transform supported by only {n_inliers} inlier "
            f"match(es); need at least {MIN_INLIER_COUNT}. Try more keypoints"
            f"or a better reference image."
        )
    _reject_implausible_transform(model_robust, np.shape(frame))

    return transform.warp(frame, model_robust.inverse)


def _image_registration(
    frame: np.ndarray, reference: np.ndarray, feat_keypoints: int = 200
) -> np.ndarray:
    """
    Aligns image 'frame' to reference image 'ref' via similarity transform.

    Parameters
    ----------
    frame: np.ndarray
        Frame to be aligned.
    reference: np.ndarray
        Frame to align to
    feat_keypoints
        Number of feature keypoints extracted for alignment

    Returns
    -------
    np.ndarray
        Aligned frame.

    Raises
    ------
    ValueError
        If a registration transform could not be reliably estimated (too few
        feature matches or too few inliers).
    """
    ref_keypoints, ref_descriptors = _extract_features(reference, feat_keypoints)
    return _register_frame_to_features(
        frame, ref_keypoints, ref_descriptors, feat_keypoints
    )
