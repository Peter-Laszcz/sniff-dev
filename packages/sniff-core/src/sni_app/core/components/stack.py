"""
Defines the Stack class and methods, and the stack history functionality.
A stack carries its data, per-frame metadata, and whole-stack metadata.

History:
Each stack stores a UUID so it can be tracked through a process network digraph.
A stack produced by a process also stores the process name, its parameters,
the UUIDs of the stacks it was made from, and how the process used them (map or reduce).
A stack with no history is an entry point and must be provided on replay.

History schema:

    {
      "process": str, # Name of process called to produce stack
      "params": dict, # parameters used to produce stack
      "inputs": list[str], # UUIDs of stacks selected to produce current stack (in order)
      "aux": dict[str,str], # (name:UUID) of auxiliary inputs (open_beam, reference, etc.)
      "mode": str ("map" | "reduce"), # function type (one-to-one/many-to-one)
      "process_id": str, # UUID of process used to produce stack
      "output_index": int, # output index of stack in previous process output.
      "output_count": int, # number of output stacks in previous process output
      "chain": list[dict["process": str, "params": dict]],
    }

"""

import datetime
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile
from astropy.io import fits

from sni_app.core.io.image import (
    ALLOWED_EXTENSIONS,
    _FITS_EXTENSIONS,
    _TIFF_EXTENSIONS,
    _get_img,
    _get_imgs_parallel,
    _write_img,
)
from sni_app.core.io.stack import (
    list_stack_frames,
    _log,
    scan_experiment_txts,
)
from sni_app.core.util.run_stats import frame_wavelengths


@dataclass
class Stack:
    """
    Holds stack data, stack metadata, and headers (metadata) of stack slices.

    Attributes
    ----------
    data : ndarray
        NumPy array of experiment stack pixel values (axis order : TYX).
    headers : list[fits.Header]
        List of FITS headers, with list index corresponding to stack slice number, derived from imported slices.
    stack_meta : dict
        Dictionary of metadata relevant to stack processing, e.g. process history, overlap correction arrays, etc.
    path : Path | None
        Path to the stack folder, where relevant. Stored as None when stack is created via other means (and therefore
        stored in memory).
    """
    data: np.ndarray
    headers: list[fits.Header]
    stack_meta: dict
    path: Optional[Path]

    @classmethod
    def from_folder(cls, folder_path: Path, progress_callback=None):
        """
        Generate stack from a folder of 2D images of matching shape.
        Also absorbs shutter/spectra data relevant to overlap correction, if available.

        Parameters
        ----------
        folder_path: Path
            Path of folder containing images. Image extensions must be in ALLOWED_EXTENSIONS to be read.
        progress_callback : Callable[[int, int], None], optional
            Called throughout loading, if provided, to update the GUI's progress bar.

        Returns
        -------
        Stack
             Populated by images in folder.
        """
        frames = list_stack_frames(folder_path)
        if not frames:
            raise ValueError(f"No readable image frames found in {folder_path}.")

        data = None
        headers = []
        n_read = 0
        expected_shape = None


        #Frames are read in order into a pre-allocated
        #float32 array, sized from the first frame's shape.
        stack_meta: dict = {}
        for idx, (img, img_info) in enumerate(_get_imgs_parallel(frames), 1):
            if progress_callback is not None:
                progress_callback(idx, len(frames))
            if expected_shape is None:
                expected_shape = img.shape
                data = np.empty((len(frames), *expected_shape), dtype=np.float32)
            elif img.shape != expected_shape:
                continue
            if data is not None:
                data[n_read] = img
            headers.append(img_info)
            n_read += 1
        wavelengths = [hdr["wlength"] for hdr in headers if "wlength" in hdr] # abbreviated for FITS header standards
        if wavelengths and len(wavelengths) == n_read:
            stack_meta["wavelengths"] = wavelengths
        if n_read < len(frames):  # skip images not matching shape of first image.
            data = data[:n_read].copy() if data is not None else None
        _log.debug(
            f"Stack at {folder_path} has shape {data.shape if data is not None else 'NULL'}"
        )

        # Search for shutter/spectra info
        run_meta = scan_experiment_txts(Path(folder_path))
        if run_meta:
            stack_meta["run_meta"] = run_meta
            _log.debug(f"Found overlap-correction data at {folder_path}") # multiple uses but simplified for new users
            spectra = run_meta.get("spectra")
            if (
                spectra is not None
                and data is not None
                and data.shape[0] == spectra.shape[0] + 1
            ):
                data = data[
                    :-1
                ]  # Drop summed image if existence is corroborated by spectra shape
                headers = headers[
                    :-1
                ]

        return cls(
            data=data if data is not None else np.empty(len(frames), dtype=np.float32),
            headers=headers,
            stack_meta=stack_meta,
            path=folder_path,
        )

    @classmethod
    def from_fits_list(
        cls,
        image_paths: list[Path],
        meta: Optional[dict] = None,
        sort: bool = False,
    ):
        """
        Load stack from a list of .fits files.
        Resultant stack has no origin path or metadata by default. Metadata can be injected.
        Optionally sort list alphabetically (to order runs) before generating stack.

        Parameters
        ----------
        image_paths : list[Path]
            List of paths to FITS files.
        meta : dict | None
            Stack metadata to inject.
        sort : bool
            Whether to sort paths alphabetically before building stack, to sort runs by slice number.

        Returns
        -------
        Stack
             Populated by images in list.
        """
        if sort:
            image_paths = sorted(image_paths)
        datas = []
        headers = []
        for data, header in _get_imgs_parallel(image_paths):
            datas.append(data)
            headers.append(header)
        return cls(
            data=np.stack(datas), headers=headers, stack_meta=(meta or {}), path=None
        )

    @classmethod
    def from_multipage_tiff(cls, image_path: Path, meta: Optional[dict] = None):
        """
        Load stack from a multipage TIFF.
        Resultant stack has no origin path, experiment headers, or metadata. Metadata can be injected.

        Parameters
        ----------
        image_path : Path
            Path to multipage TIFF.
        meta : dict | None
            Metadata to inject.

        Returns
        -------
        Stack
             Populated by TIFF content.
        """
        data, header = _get_img(image_path)
        return cls(
            data=data,
            headers=[header.copy() for _ in range(data.shape[0])],
            stack_meta=(meta or {}),
            path=None,
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        headers: Optional[list[fits.Header]] = None,
        stack_meta: Optional[dict] = None,
    ):
        """
        Generate stack from a NumPy array, fitting to compatible shape if required.
        Metadata and headers can be injected (path remains None).

        Parameters
        ----------
        array : np.ndarray
            NumPy array of experiment stack pixel values. Handles between 1 and 3 dimensions (inclusive)
        headers : list[fits.Header] | None
            Headers per slice (acquisition frame)
        stack_meta : dict | None
            Stack metadata.

        Returns
        -------
        Stack
            A stack containing the given parameters
        """
        data = np.asarray(array, dtype=np.float32)
        if data.ndim == 2:  # single image slice
            data = data[None, :, :]
        elif data.ndim == 1:  # profile plot
            data = data[:, None, None]
        n = int(data.shape[0])
        if not headers:
            headers = [fits.Header() for _ in range(n)]
        elif len(headers) != n:
            raise ValueError(
                f"Mismatch between number of headers ({len(headers)}) and "
                f"number of frames ({n})"
            )

        return cls(
            data=data,
            headers=headers,
            stack_meta=dict(stack_meta or {}),
            path=None,
        )

    ##########
    # SAVING #
    ##########

    def save_stack(self, file_path, save_dir: Path, overwrite=False):
        """
        Write stack to a given directory, in FITS (default) or TIFF format.
        FITS saves slicewise (each slice is saved as a separate image). TIFF saves as a single multipage file.
        Output filename is of the form {file_name}_{frame_index}.{extension}

        Parameters
        ----------
        file_path : Path
            Image file path (if file extension not included, saves FITS by default.)
        save_dir : Path
            Path to directory where images will be saved.
        overwrite : bool
            Whether to overwrite existing files.

        Returns
        -------
        None

        """
        stem, extension = os.path.splitext(str(file_path))
        extension = extension.lower()
        path = Path.joinpath(save_dir, str(file_path))
        if not extension:
            extension = ".fits"
        headers = [h.copy() for h in self.headers]
        if "wavelengths" in self.stack_meta.keys():
            for wavelength, header in zip(self.stack_meta["wavelengths"], headers):
                header["wlength"] = wavelength # allows reimport of wavelengths into other workflows
        n_slices = int(self.data.shape[0])
        if extension in _FITS_EXTENSIONS:
            pad = max(4, len(str(max(0, n_slices - 1))))
            for i in range(n_slices):
                frame_name = f"{stem}_{i:0{pad}d}" # pads with zeroes (min 4 digits for readability)
                _write_img(
                    (self.data[i], headers[i]),
                    frame_name,
                    save_dir,
                    overwrite,
                    extension,
                )
        elif extension in _TIFF_EXTENSIONS:
            data = self.data
            if data.ndim == 2:
                data = data[None, :, :]
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path.name} exists (enable overwrite)")
            tifffile.imwrite(str(path), data)
        else:
            if extension in ALLOWED_EXTENSIONS:
                raise ValueError(f"Cannot write stack to extension: {extension}")
            raise ValueError(f"Unrecognized file extension: {extension}")

    def save_energies(
        self,
        dst_dir: Path,
        folder_name: str = "",
        img_name: str = "transmission",
        overwrite: bool = False,
        he_le: tuple[list[int], list[int]] = ([15, 30], [30, 50]),
    ):
        """
        Save stack data as two images holding averages of given short and long wavelength ranges.
        FITS header is taken from median slice in range.

        Parameters
        ----------
        dst_dir : Path
            Path to directory where high/low energy folders will be produced.
        folder_name : str
            Name of parent folder of high/low folder split.
        img_name : str
           String to write to image names. Saved images have format
           {high or low}_energy_{img_name}.fits
        overwrite : bool
            Whether to overwrite existing files.
        he_le : tuple[list[int], list[int]]
            Tuple containing high and low energy ranges.

        Returns
        -------
        None

        """

        folder_dir = folder_name if folder_name else "stack_out"
        if self.data.shape[0] > 2:
            he_avg = np.nanmean(self.data[he_le[0][0] : he_le[0][1]], axis=0)
            le_avg = np.nanmean(self.data[he_le[1][0] : he_le[1][1]], axis=0)
            he_hdr = self.headers[int((he_le[0][0] + (he_le[0][1]) // 2))]
            le_hdr = self.headers[int((he_le[1][0] + (he_le[1][1]) // 2))]
        else:
            he_avg, le_avg = (
                self.data[
                    0,
                    :,
                    :,
                ],
                self.data[
                    1,
                    :,
                    :,
                ],
            )
            he_hdr, le_hdr = self.headers[0], self.headers[1]

        path_he = Path.joinpath(dst_dir, folder_dir, "HE_stack")
        path_le = Path.joinpath(dst_dir, folder_dir, "LE_stack")
        os.makedirs(path_he, exist_ok=True)
        os.makedirs(path_le, exist_ok=True)
        file_name_he = f"high_energy_{img_name}"
        file_name_le = f"low_energy_{img_name}"
        he_hdr, le_hdr = (
            he_hdr.copy(),
            le_hdr.copy(),
        )  # leave the stack's own headers be
        for hdr in (he_hdr, le_hdr):
            hdr["HISTORY"] = f"Image saved: {datetime.datetime.now()}"
        _write_img((he_avg, he_hdr), file_name_he, base_dir=path_he, overwrite=overwrite)
        _write_img((le_avg, le_hdr), file_name_le, base_dir=path_le, overwrite=overwrite)

    ############
    # METADATA #
    ############

    def _meta(self) -> dict:
        """
        Return the stack's metadata dict, creating one if None.

        Returns
        -------
        dict
            Stack's metadata dictionary.
        """
        meta = getattr(self, "stack_meta", None)
        if not isinstance(meta, dict):
            meta = {}
            self.stack_meta = {}
        return meta

    def display_name(self) -> str:
        """
        Return the stack's display name (used in GUI). Return (path or "stack") if not set.

        Returns
        -------
        str
        """
        name = self._meta().get("display_name")
        if name:
            return str(name)
        return Path(self.path).name if self.path is not None else "stack"

    def run_meta_data(self) -> Optional[dict[str, np.ndarray]]:
        """
        Return array dictionary of experiment spectra/shutter arrays (if found).

        Returns
        -------
        dict[str, np.ndarray] | None
        keys:("shutter_count", "shutter_times", "spectra")

        """
        if "run_meta" in self.stack_meta:
            return self.stack_meta["run_meta"]
        return None

    def times_of_flight(self) -> Optional[np.ndarray]:
        """
        Return the stack's time-of-flight array (framewise). Requires "spectra_times" data.

        Returns
        -------
        np.ndarray | None
            Framewise ToF, or None when the stack has no spectra data.
        """
        times = self._meta().get("spectra_times")
        if times is None:
            spectra = (self.run_meta_data() or {}).get("spectra")
            if spectra is None:
                return None
            spectra = np.asarray(spectra)
            times = spectra[:, 0]
        times = np.asarray(times, dtype=np.float64).ravel()
        return times if times.size else None

    def set_times_of_flight(self, times: np.ndarray) -> None:
        """
        Record per-frame acquisition times on the stack.

        Parameters
        ----------
        times : np.ndarray
            framewise times.

        Returns
        -------
        None
        """
        meta = self._meta()
        meta.pop("spectra_times", None)
        meta.update({"spectra_times": times})

    def wavelengths(
        self,
        delay: float = 0.0,
        collimation_distance: float = 0.0,
        apply_delay: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Return per-slice neutron wavelengths, in angstroms.
        Derived from the stack's spectra_times and beamline geometry (experiment dependent).
        Note that this is different to accessing wavelengths in stack_meta, which should be accessed instead if trusted.
        Running this function overwrites previous stack_meta wavelengths.

        Parameters
        ----------
        delay : float
            Acquisition delay (useful for synchronising long and short wavelength runs).
        collimation_distance : float
            Flight-path length, in meters.
        apply_delay : bool
            Whether to apply the delay (i.e. for wavelength disparity).

        Returns
        -------
        np.ndarray | None
            One wavelength per frame, or None when the stack carries no times
            or parameters are malformed.
        """
        times = self.times_of_flight()
        if times is None or collimation_distance <= 0:
            return None
        wavelengths = frame_wavelengths(times, delay, collimation_distance, apply_delay)
        self.stack_meta["wavelengths"] = wavelengths
        return  wavelengths

    ####################
    # ANALYSIS RESULTS #
    ####################

    def record_analysis_results(self, results: dict):
        """
        Append non-array results of post-processing computation (e.g. ROI functions) to stack's metadata.

        Parameters
        ----------
        results : dict
            Dictionary of analysis results.

        Returns
        -------
        None
        """
        if not results:
            return
        self._meta()["analysis_results"] = self._meta()["analysis_results"] & dict(results)

    def analysis_results(self) -> dict:
        """
        Return non-array results of post-processing computation stored on stack.

        Returns
        -------
        dict
        Dictionary of analysis results.

        """
        results = self._meta().get("analysis_results")
        return dict(results) if isinstance(results, dict) else {}

    def analysis_results_text(self) -> str:
        """
        Produce readable string of a stack's post-processing computation results.

        Returns
        -------
        str
        String rendering of analysis results
        """
        results = self.analysis_results()
        if not results:
            return "(no analysis results)"
        lines = []
        for key, value in results.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6g}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    ###########
    #   UUID  #
    ###########

    def assign_stack_uuid(self) -> str:
        """
        Assign a new UUID to the stack. Replaces pre-existing UUIDs.

        Returns
        -------
        str
            The new UUID.
        """
        stack_uuid = uuid.uuid4().hex

        self._meta()["stack_uuid"] = stack_uuid
        return stack_uuid

    def stack_uuid(self) -> Optional[str]:
        """
        Return the stack's UUID, or None if it has not been assigned one.

        Returns
        -------
        str | None
        """
        return self._meta().get("stack_uuid")

    def robust_stack_uuid(self) -> str:
        """
        Return stack UUID, assigning a fresh one if it has none.

        Returns
        -------
        str
        """
        sid = self._meta().get("stack_uuid")
        if not sid:
            sid = self.assign_stack_uuid()
        return sid

    ########################
    #  PROCESS DERIVATION  #
    ########################

    def get_history(self) -> Optional[dict]:
        """
        Return the stack's history record, or None for an entry point/ auxiliary stack.

        Returns
        -------
        dict | None
            Stack history record, if applicable.
        """
        history = self._meta().get("history")
        return history if isinstance(history, dict) else None

    def is_entry_point(self) -> bool:
        """
        Return whether the stack is an entry point of a workflow.

        Returns
        -------
        bool
            True when nothing produced this stack from other stacks.
        """
        history = self.get_history()
        return not history or (
            not history.get("inputs") and not (history.get("aux") or {})
        )

    def parent_ids(self) -> List[str]:
        """
        Return the UUIDs of the stacks this stack was derived from.

        Returns
        -------
        list of str
            Primary inputs first, then auxiliary inputs. Empty for an entry point.
        """
        history = self.get_history() or {}
        return list(history.get("inputs") or []) + list(
            (history.get("aux") or {}).values()
        )

    def process_history(self) -> list:
        """
        Return the ordered chain of processes that produced this stack.

        Returns
        -------
        list of dict
            One {"process": str, "params": dict} entry per step, oldest
            first, ending with the process that produced this stack. Empty for
            an entry point.
        """
        history = self.get_history()
        if not history:
            return []
        chain = history.get("chain")
        return list(chain) if isinstance(chain, list) else []

    def process_history_string(self) -> str:
        """
        Readable string form of a stack's process history (used for GUI).

        Returns
        -------
        str

        """
        hist = self.process_history()
        if not hist:
            return "(no recorded processes)"
        lines = []
        for i, proc in enumerate(hist, 1):
            params = proc.get("params", {}) or {}
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(
                f"{i}. {proc.get('process', '?')}"
                + (f"  ({param_str})" if param_str else "")
            )
        return "\n".join(lines)


def record_derivation(
    outputs: list["Stack"],
    process: str,
    params: Optional[dict],
    inputs: list["Stack"],
    aux: Optional[dict[str, "Stack"]] = None,
    mode: str = "map",
) -> List["Stack"]:
    """
    Record workflow history to stacks produced by a process.
    Stacks inherit previous history if applicable with aforementioned process appended.
    #TODO: This must be called by plugin processes in the future to enable workflow integration

    Parameters
    ----------
    outputs : list[Stack]
        Stacks produced by the process, in order.
    process : str
        Name of process as defined in process registry.
    params : mapping or None
        Process parameters. Must be serialisable, or else sanitised via sanitise_params.
    inputs : list[Stack]
        Process input stacks, in order.
    aux : dict[str, Stack], optional
        Named auxiliary input stacks (e.g. {"open_beam": ob}) fed into process.
    mode : {"map", "reduce"}
        Output/input relationship (one-to-one/many-to-one).


    Returns
    -------
    list of Stack
        updated output stacks.
    """
    from sni_app.core.components.workflow import sanitise_params

    input_ids = [s.robust_stack_uuid() for s in inputs]
    aux_ids: Dict[str, str] = {
        str(role): s.robust_stack_uuid() for role, s in (aux or {}).items()
    }
    clean_params = sanitise_params(params)
    call_id = uuid.uuid4().hex

    n = len(outputs)

    for i, out in enumerate(outputs):
        if mode == "reduce":
            these_inputs = list(input_ids)
            base_source = inputs[0] if inputs else None
        else:  # one-to-one
            these_inputs = [input_ids[i]] if i < len(input_ids) else list(input_ids)
            base_source = (
                inputs[i] if i < len(inputs) else (inputs[0] if inputs else None)
            )
        inherited = (
            base_source.process_history() if base_source is not None else []
        )  # inherit flattened history

        out.assign_stack_uuid()  # replace any id copied from a parent's meta
        out._meta()["history"] = {
            "process": process,
            "params": clean_params,
            "inputs": these_inputs,
            "aux": dict(aux_ids),
            "mode": mode,
            "call_id": call_id,
            "output_index": i,
            "output_count": n,
            "chain": inherited + [{"process": process, "params": dict(clean_params)}],
        }

    return outputs
