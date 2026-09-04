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
SNIFF : Stack Stitcher
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from sni_app.gui.shared import BTN_STYLE_RED, BTN_STYLE_RUN, popup

from sni_app.core.components.stack import Stack
from sni_app.core.process.img_processes import _robust_percentile_limits
from sni_app.core.process.roi_processes import clamp_roi_to_stack, roi_profile
from sni_app.core.process.stack_processes import stack_stitching
from sni_app.core.util import logger

################
# GRAPH CONFIG #
################

# profile plot colours #TODO : consider colours for accessibility
_SHORT_PEN = "#4fc3f7"  # light blue
_LONG_PEN = "#ce93d8"  # purple
_COMBINED_PEN = "#ff5555"  # red

pg.setConfigOption("background", "#2b2b2b")  # dark background for image views
pg.setConfigOption("foreground", "#dddddd")  # light axis labels / tick marks
pg.setConfigOption("imageAxisOrder", "row-major")


def _gamma_lut(gamma: float) -> np.ndarray:
    """Greyscale lookup table applying gamma correction."""
    x = np.linspace(0.0, 1.0, 256)
    y = np.clip(np.power(x, 1.0 / max(float(gamma), 1e-6)), 0.0, 1.0)
    g = (y * 255.0).astype(np.uint8)
    return np.column_stack([g, g, g, np.full(256, 255, dtype=np.uint8)])


def _stack_name(stack: Stack) -> str:
    """Human-readable stack name (from its path or '(in-memory)')."""
    path = getattr(stack, "path", None)
    if path is None:
        return "(in-memory)"
    return Path(path).name or str(path)


class StitcherDialog(QtWidgets.QDialog):
    stitch_finished = QtCore.pyqtSignal(
        object
    )  # emitted with the stitched stack when finished.

    def __init__(
        self,
        short: Stack,
        long: Stack,
        delay: float,
        collimation_distance: float,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """
        Build stitcher dialog for a given short/long stack pair.

        Parameters
        ----------
        short : Stack
            The short-wavelength stack.
        long : Stack
            The long-wavelength stack.
        parent : QtWidgets.QWidget, optional
            Parent widget.
        """
        super().__init__(parent)
        self.log = logger.setup_logger()

        self.short = short
        self.long = long
        self.short_wavelengths = short.wavelengths(delay, collimation_distance, False)
        self.long_wavelengths = long.wavelengths(delay, collimation_distance, True)
        self.result_stack: Optional[Stack] = None
        self.delay = delay
        self.collimation_distance = collimation_distance

        self.roi_xywh: tuple[int, int, int, int] = clamp_roi_to_stack(
            (0, 0, 100, 100), short
        )
        self._current_frame = {"short": 0, "long": 0}
        self._updating_spins = False
        self.views: Dict[str, Dict[str, object]] = {}

        self.setWindowTitle("Stack Stitcher (Short + Long)")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(840, 574)

        self._build_ui()
        self._connect_signals()
        self._init_after_load()

    ######
    # UI #
    ######

    def _build_ui(self) -> None:
        """Assemble the GUI layout."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # image views
        img_row = QtWidgets.QHBoxLayout()
        img_row.addLayout(self._build_image_column("short", "Short", "y"))
        img_row.addLayout(self._build_image_column("long", "Long", "c"))
        root.addLayout(img_row, stretch=3)

        # frame-range controls
        root.addLayout(self._build_controls_row())

        # viewer display controls.
        root.addWidget(self._build_display_group())

        # profile plots
        root.addWidget(self._build_profile_plot(), stretch=2)

        # cancel/finish dialog buttons.
        root.addLayout(self._build_button_row())

    def _build_image_column(
        self, view: str, title: str, roi_colour: str
    ) -> QtWidgets.QVBoxLayout:
        """Build one titled pyqtgraph image view with an ROI overlay and slider."""
        col = QtWidgets.QVBoxLayout()

        image_title_lbl = QtWidgets.QLabel(
            f"{title} – {_stack_name(getattr(self, view))}"
        )
        image_title_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        image_title_lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        col.addWidget(image_title_lbl)

        glw = pg.GraphicsLayoutWidget()
        glw.setMinimumHeight(240)
        view_box = glw.addViewBox(lockAspect=True)
        img = pg.ImageItem()
        view_box.addItem(img)
        col.addWidget(glw, stretch=1)

        # right hand ROI is only updated via sync button.
        x0, y0, w, h = self.roi_xywh
        roi = pg.RectROI(
            pos=[x0, y0],
            size=[w, h],
            pen=pg.mkPen(roi_colour, width=2),
            hoverPen=pg.mkPen(roi_colour, width=3),
            movable=(view == "short"),
        )
        if view == "short":
            roi.addScaleHandle([1, 0.5], [0, 0.5])
            roi.addScaleHandle([0, 0.5], [1, 0.5])
            roi.addScaleHandle([0.5, 0], [0.5, 1])
            roi.addScaleHandle([0.5, 1], [0.5, 0])
        else:  # long ROI
            while roi.handles:
                roi.removeHandle(0)  # no ROI handles for you >:)
        view_box.addItem(roi)

        roi_text = pg.TextItem("ROI", color=roi_colour, anchor=(0, 1))
        roi_text.setPos(x0, y0)
        view_box.addItem(roi_text)

        self.views[view] = {
            "glw": glw,
            "view_box": view_box,
            "img": img,
            "roi": roi,
            "roi_text": roi_text,
            "image_title_lbl": image_title_lbl,
            "levels": (0.0, 1.0),
        }
        col.addLayout(self._build_slider_row(view))
        return col

    def _build_slider_row(self, view: str) -> QtWidgets.QHBoxLayout:
        """Create a labelled frame slider."""
        row = QtWidgets.QHBoxLayout()

        lbl = QtWidgets.QLabel("Frame")
        lbl.setFixedWidth(44)
        lbl.setStyleSheet("font-size: 10px;")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        row.addWidget(lbl)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(0)
        slider.setValue(0)
        slider.valueChanged.connect(lambda v, k=view: self._show_frame(k, v))
        row.addWidget(slider)

        frame_lbl = QtWidgets.QLabel("0")
        frame_lbl.setFixedWidth(34)
        frame_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        row.addWidget(frame_lbl)

        self.views[view]["slider"] = slider
        self.views[view]["frame_lbl"] = frame_lbl
        return row

    def _build_controls_row(self) -> QtWidgets.QHBoxLayout:
        """Build the frame-range spin boxes plus the sync/recompute buttons."""
        row = QtWidgets.QHBoxLayout()

        short_box = QtWidgets.QGroupBox("Short range [short_lower:short_upper)")
        short_layout = QtWidgets.QHBoxLayout(short_box)
        self.short_lower = QtWidgets.QSpinBox()
        self.short_upper = QtWidgets.QSpinBox()
        short_layout.addWidget(QtWidgets.QLabel("short_lower"))
        short_layout.addWidget(self.short_lower)
        short_layout.addWidget(QtWidgets.QLabel("short_upper"))
        short_layout.addWidget(self.short_upper)
        row.addWidget(short_box)

        long_box = QtWidgets.QGroupBox("Long range [long_lower:long_upper)")
        long_layout = QtWidgets.QHBoxLayout(long_box)
        self.long_lower = QtWidgets.QSpinBox()
        self.long_upper = QtWidgets.QSpinBox()
        long_layout.addWidget(QtWidgets.QLabel("long_lower"))
        long_layout.addWidget(self.long_lower)
        long_layout.addWidget(QtWidgets.QLabel("long_upper"))
        long_layout.addWidget(self.long_upper)
        row.addWidget(long_box)

        self.roi_sync_btn = QtWidgets.QPushButton("Sync ROI to Long")
        self.recalc_btn = QtWidgets.QPushButton("Recompute Profiles")
        row.addWidget(self.roi_sync_btn)
        row.addWidget(self.recalc_btn)
        row.addStretch()
        return row

    def _build_display_group(self) -> QtWidgets.QGroupBox:
        """Build the wrapper for contrast controls."""
        group_box = QtWidgets.QGroupBox("Display")
        layout = QtWidgets.QHBoxLayout(group_box)
        layout.addWidget(self._make_display_controls("short", "Short"))
        layout.addWidget(self._make_display_controls("long", "Long"))
        return group_box

    def _make_display_controls(self, view: str, title: str) -> QtWidgets.QGroupBox:
        """
        Build one view's min/max/gamma/auto-contrast controls.

        Parameters
        ----------
        view : str
            View key ("short" / "long"); the created widgets are stored
            back into self.views[view].
        title : str
            Group-box title.

        Returns
        -------
        QtWidgets.QGroupBox
            The assembled controls group box.
        """
        group_box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QHBoxLayout(group_box)

        auto_btn = QtWidgets.QPushButton("Auto Contrast")
        layout.addWidget(auto_btn)

        min_spin = QtWidgets.QDoubleSpinBox()
        max_spin = QtWidgets.QDoubleSpinBox()
        for spin_val in (min_spin, max_spin):
            spin_val.setDecimals(4)
            spin_val.setRange(-1e12, 1e12)
            spin_val.setSingleStep(1.0)
            spin_val.setKeyboardTracking(False)
        layout.addWidget(QtWidgets.QLabel("Min"))
        layout.addWidget(min_spin)
        layout.addWidget(QtWidgets.QLabel("Max"))
        layout.addWidget(max_spin)

        gamma_spin = QtWidgets.QDoubleSpinBox()
        gamma_spin.setDecimals(2)
        gamma_spin.setRange(0.2, 3.0)
        gamma_spin.setSingleStep(0.1)
        gamma_spin.setValue(1.0)
        gamma_spin.setKeyboardTracking(False)
        layout.addWidget(QtWidgets.QLabel("Gamma"))
        layout.addWidget(gamma_spin)

        self.views[view].update(
            {
                "auto_btn": auto_btn,
                "min_spin": min_spin,
                "max_spin": max_spin,
                "gamma_spin": gamma_spin,
            }
        )
        return group_box

    def _build_profile_plot(self) -> pg.PlotWidget:
        """Create the pyqtgraph plot widget used for the ROI z-profiles."""
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Wavelength (Å)")
        self._plot.setLabel("left", "Mean ROI intensity")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._legend = self._plot.addLegend(offset=(-10, 10))
        self._plot.setMinimumHeight(180)
        return self._plot

    def _build_button_row(self) -> QtWidgets.QHBoxLayout:
        """Build the bottom cancel/finish button row."""
        row = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(BTN_STYLE_RED)
        self.finish_btn = QtWidgets.QPushButton("Finish")
        self.finish_btn.setStyleSheet(BTN_STYLE_RUN)
        self.finish_btn.setDefault(True)
        row.addStretch()
        row.addWidget(self.cancel_btn)
        row.addWidget(self.finish_btn)
        return row

    ###########
    # SIGNALS #
    ###########

    def _connect_signals(self) -> None:
        """Wire buttons, spin boxes, the ROI and per-view controls to their handlers."""
        self.recalc_btn.clicked.connect(self.recompute)
        self.roi_sync_btn.clicked.connect(self.sync_roi)
        self.finish_btn.clicked.connect(self._on_finish)
        self.cancel_btn.clicked.connect(self.reject)

        for sp in (
            self.short_lower,
            self.short_upper,
            self.long_lower,
            self.long_upper,
        ):
            sp.valueChanged.connect(self.recompute)

        self.views["short"]["roi"].sigRegionChanged.connect(self._on_roi_moved)

        for view in ("short", "long"):
            v = self.views[view]
            v["auto_btn"].clicked.connect(lambda _=False, k=view: self.auto_contrast(k))
            v["min_spin"].valueChanged.connect(
                lambda _v, k=view: self._apply_display_controls(k)
            )
            v["max_spin"].valueChanged.connect(
                lambda _v, k=view: self._apply_display_controls(k)
            )
            v["gamma_spin"].valueChanged.connect(
                lambda _v, k=view: self._apply_display_controls(k)
            )

    ################
    # INITIALISERS #
    ################

    def _init_after_load(self) -> None:
        """Initialise ranges, sliders, contrast and profiles once both stacks are set."""
        self.log.info(
            f"Stitcher loaded: short {tuple(self.short.data.shape)}, "
            f"long {tuple(self.long.data.shape)}"
        )
        self._set_range_spinboxes()
        for view in ("short", "long"):
            stack = getattr(self, view)
            n_slices = int(stack.data.shape[0])
            self.views[view]["slider"].setMaximum(max(0, n_slices - 1))
            pix_val_min, pix_val_max = _robust_percentile_limits(stack.data)
            self.views[view]["levels"] = (pix_val_min, pix_val_max)
            self._set_display_spins(view, pix_val_min, pix_val_max, 1.0)
            self._show_frame(view, 0)
        self.recompute()

    def _set_range_spinboxes(self) -> None:
        """Set the short_lower/short_upper/long_lower/long_upper ranges to span each stack."""
        short_max_frame = int(self.short.data.shape[0])
        long_max_frame = int(self.long.data.shape[0])
        self._updating_spins = True
        try:
            self.short_lower.setRange(0, short_max_frame)
            self.short_upper.setRange(0, short_max_frame)
            self.long_lower.setRange(0, long_max_frame)
            self.long_upper.setRange(0, long_max_frame)
            self.short_lower.setValue(0)
            self.short_upper.setValue(short_max_frame)
            self.long_lower.setValue(0)
            self.long_upper.setValue(long_max_frame)
        finally:
            self._updating_spins = False

    ##################
    # ROI OPERATIONS #
    ##################

    def _roi_to_xywh(self, roi: pg.RectROI, stack: Stack) -> Tuple[int, int, int, int]:
        """Convert a RectROI (pos=[x,y], size=[w,h]) to a clamped xywh tuple."""
        pos = roi.pos()
        size = roi.size()
        x0 = int(round(pos.x()))
        y0 = int(round(pos.y()))
        w = max(1, int(round(size.x())))
        h = max(1, int(round(size.y())))
        return clamp_roi_to_stack((x0, y0, w, h), stack)

    def _current_roi_xywh(self) -> Tuple[int, int, int, int]:
        """Return (and store) the short-view ROI as a clamped (x, y, w, h) tuple."""
        roi = self._roi_to_xywh(self.views["short"]["roi"], self.short)
        self.roi_xywh = roi
        return roi

    def _on_roi_moved(self) -> None:
        """Reposition the ROI label and recompute profiles when the ROI is dragged."""
        roi = self.views["short"]["roi"]
        pos = roi.pos()
        self.views["short"]["roi_text"].setPos(pos.x(), pos.y())
        self.recompute()

    def sync_roi(self) -> None:
        """Copy the short-view ROI (clamped to the long stack) onto the long view."""
        x0, y0, w, h = clamp_roi_to_stack(self._current_roi_xywh(), self.long)
        long_roi = self.views["long"]["roi"]
        long_roi.setPos([x0, y0], update=False)
        long_roi.setSize([w, h])
        self.views["long"]["roi_text"].setPos(x0, y0)
        self.recompute()

    ################################
    # DISPLAY AND DISPLAY CONTROLS #
    ################################
    def _show_frame(self, view: str, idx: int) -> None:
        """
        Display indexed frame of a view, reapplying its contrast levels for viewing ease.

        Parameters
        ----------
        view : str
            View key ("short" / "long").
        idx : int
            Frame index to show (clamped to the stack).
        """
        stack = getattr(self, view)
        n_slices = int(stack.data.shape[0])
        idx = max(0, min(int(idx), n_slices - 1))
        self._current_frame[view] = idx

        frame_view = self.views[view]
        frame_view["img"].setImage(np.asarray(stack.data[idx]), autoLevels=False)
        self._apply_levels(view)
        frame_view["frame_lbl"].setText(str(idx))

    def _set_display_spins(
        self, view: str, contrast_min: float, contrast_max: float, gamma: float
    ) -> None:
        """
        Set a view's min/max/gamma spin boxes and cache levels.

        Parameters
        ----------
        view : str
            View key ("short" / "long").
        contrast_min, contrast_max : float
            Contrast limits to display and cache.
        gamma : float
            Gamma value to display.
        """
        stack_view = self.views[view]
        self._updating_spins = True
        try:
            stack_view["min_spin"].setValue(float(contrast_min))
            stack_view["max_spin"].setValue(float(contrast_max))
            stack_view["gamma_spin"].setValue(float(gamma))
        finally:
            self._updating_spins = False
        stack_view["levels"] = (float(contrast_min), float(contrast_max))

    def _apply_levels(self, view: str) -> None:
        """Apply a view's contrast levels and gamma LUT to its image item."""
        stack_view = self.views[view]
        contrast_min, contrast_max = stack_view["levels"]
        gamma = float(stack_view["gamma_spin"].value())
        try:
            stack_view["img"].setLookupTable(_gamma_lut(gamma))
            stack_view["img"].setLevels([contrast_min, contrast_max])
        except Exception as exc:
            self.log.warning(f"Failed applying {view} display levels: {exc}")

    def _apply_display_controls(self, view: str) -> None:
        """
        Push a view's spin-box values onto its stored levels and redraw.

        Parameters
        ----------
        view : str
            View key ("short" / "long").
        """
        if self._updating_spins:
            return
        stack_view = self.views[view]
        contrast_min = float(stack_view["min_spin"].value())
        contrast_max = float(stack_view["max_spin"].value())
        if contrast_max <= contrast_min:
            contrast_max = contrast_min + 1e-6
            self._updating_spins = True
            try:
                stack_view["max_spin"].setValue(contrast_max)
            finally:
                self._updating_spins = False
        stack_view["levels"] = (contrast_min, contrast_max)
        self._apply_levels(view)

    def auto_contrast(self, view: str) -> None:
        """
        Set a view's contrast from its currently displayed frame.

        Parameters
        ----------
        view : str
            View key ("short" / "long").
        """
        stack = getattr(self, view)
        idx = self._current_frame.get(view, 0)
        idx = max(0, min(idx, stack.data.shape[0] - 1))
        pix_val_min, pix_val_max = _robust_percentile_limits(stack.data[idx])
        gamma = float(self.views[view]["gamma_spin"].value())
        self._set_display_spins(view, pix_val_min, pix_val_max, gamma)
        self._apply_levels(view)

    #################
    # PROFILE PLOTS #
    #################

    @staticmethod
    def _clamp_range(a: int, b: int, n: int) -> Tuple[int, int]:
        """
        Clamp a [a, b) range to [0, n].

        Parameters
        ----------
        a, b : int
            Range endpoints (swapped if b < a).
        n : int
            Upper bound (inclusive) to clamp to.

        Returns
        -------
        Tuple[int, int]
            The clamped and ordered (a, b).
        """
        a = max(0, min(int(a), n))
        b = max(0, min(int(b), n))
        if b < a:
            a, b = b, a
        return a, b

    def recompute(self) -> None:
        """Recompute the short/long/combined ROI profiles and redraw the plot."""
        if self._updating_spins:
            return

        roi_short = self._current_roi_xywh()
        roi_long = clamp_roi_to_stack(roi_short, self.long)
        short_prof = roi_profile(roi_short, self.short)
        long_prof = roi_profile(roi_long, self.long)

        s0, s1 = self._clamp_range(
            self.short_lower.value(), self.short_upper.value(), len(short_prof)
        )
        l0, l1 = self._clamp_range(
            self.long_lower.value(), self.long_upper.value(), len(long_prof)
        )
        combined = np.concatenate([short_prof[s0:s1], long_prof[l0:l1]])

        self._plot_profiles(
            short_prof,
            long_prof,
            combined,
            (s0, s1),
            (l0, l1),
        )

    def _plot_profiles(
        self,
        short_prof: np.ndarray,
        long_prof: np.ndarray,
        combined: np.ndarray,
        s_range: Tuple[int, int],
        l_range: Tuple[int, int],
    ) -> None:
        """
        Draw the short/long/combined profiles with shaded selected ranges.

        Each profile is plotted against its stack's wavelengths where those
        could be derived, and against frame index otherwise.

        Parameters
        ----------
        short_prof, long_prof : np.ndarray
            Per-frame ROI mean profiles for each stack.
        combined : np.ndarray
            The concatenated (stitched) preview profile.
        s_range, l_range : Tuple[int, int]
            Selected (start, stop) ranges shaded on each stack's profile.
        """
        self._legend.clear()
        self._plot.clear()
        s0, s1 = s_range
        l0, l1 = l_range

        wavelength_axis = (
            self.short_wavelengths is not None and self.long_wavelengths is not None
        )
        x_short = (
            self.short_wavelengths if wavelength_axis else np.arange(len(short_prof))
        )
        x_long = self.long_wavelengths if wavelength_axis else np.arange(len(long_prof))
        self._plot.setLabel("bottom", "Wavelength (Å)" if wavelength_axis else "Frame")

        if wavelength_axis:
            x_combined = np.concatenate([x_short[s0:s1], x_long[l0:l1]])
        else:
            x_combined = np.arange(len(combined))

        if len(short_prof):
            self._plot.plot(
                x_short,
                short_prof,
                pen=pg.mkPen(_SHORT_PEN, width=1.5),
                name="Short (ROI mean)",
            )
            region = pg.LinearRegionItem(
                values=(x_short[s0 - 1], x_short[s1 - 1]),
                brush=pg.mkBrush((79, 195, 247, 45)),
                movable=False,
            )
            region.setZValue(-10)
            self._plot.addItem(region)

        if len(long_prof):
            self._plot.plot(
                x_long,
                long_prof,
                pen=pg.mkPen(_LONG_PEN, width=1.5),
                name="Long (ROI mean)",
            )
            region = pg.LinearRegionItem(
                values=(x_long[l0 - 1], x_long[l1 - 1]),
                brush=pg.mkBrush((206, 147, 216, 45)),
                movable=False,
            )
            region.setZValue(-10)
            self._plot.addItem(region)

        if len(combined):
            out = sorted(zip(x_combined, combined))
            seam = s1 - s0
            self._plot.plot(
                [x[0] for x in out],
                [x[1] for x in out],
                pen=pg.mkPen(_COMBINED_PEN, width=1.5),
                name=f"Combined preview (seam @ {seam})",
            )

    ########################
    # STITCHING OPERATIONS #
    ########################

    def _build_stitched_stack(self) -> Optional[Stack]:
        """
        Stitch the selected short and long frame ranges into one stack.

        Returns
        -------
        Stack or None
            The stitched stack, or None if the inputs are incompatible or
            the selection is empty.
        """
        s0, s1 = self._clamp_range(
            self.short_lower.value(), self.short_upper.value(), self.short.data.shape[0]
        )
        l0, l1 = self._clamp_range(
            self.long_lower.value(), self.long_upper.value(), self.long.data.shape[0]
        )
        try:
            out = stack_stitching(
                self.short,
                self.long,
                (s0, s1),
                (l0, l1),
                self.delay,
                self.collimation_distance,
            )[0]
        except ValueError as exc:
            popup(self, "Cannot stitch", str(exc))
            return None

        self.log.info(
            f"Stitched stack: short[{s0}:{s1}] + long[{l0}:{l1}] -> "
            f"{tuple(out.data.shape)}"
        )
        return out

    def _on_finish(self) -> None:
        """Build the stitched stack; on success store/emit it and accept the dialog."""
        stitched = self._build_stitched_stack()
        if stitched is None:
            return
        self.result_stack = stitched
        self.stitch_finished.emit(stitched)
        self.accept()


#############
#  RUNNING  #
#############


def stitch_stacks(
    short: Stack,
    long: Stack,
    delay: float,
    collimation_distance: float,
    parent: Optional[QtWidgets.QWidget] = None,
) -> Optional[Stack]:
    """
    Run the stitcher dialog with input stacks and return the stitched Stack (or None if cancelled).
    """
    dlg = StitcherDialog(short, long, delay, collimation_distance, parent)
    if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        return dlg.result_stack
    return None
