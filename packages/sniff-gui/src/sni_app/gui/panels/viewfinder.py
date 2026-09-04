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
Stack viewfinder and ROI plotter.
"""

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from sni_app.gui.shared import dspin, hbox, label, vbox

from sni_app.core.components.stack import Stack
from sni_app.core.process.roi_processes import compute_roi_stats


# Lineage name of the process that turns raw counts into a transmission ratio.
NORMALISATION_PROCESS = "Normalisation"

# What a stack's z-profile is measuring, once a process has changed its unit.
# The most recent of these in a stack's history wins, so a coefficient computed
# from normalised data reads as a coefficient rather than a transmission.
Z_AXIS_LABELS = {
    NORMALISATION_PROCESS: "Transmission",
    "Attenuation Coefficient": "Attenuation coefficient (cm⁻¹)",
    "Total Microscopic Cross Section": "Total microscopic cross section (barns)",
    "Hydrogen Cross Section": "Hydrogen cross section (barns)",
}


# The two detail tabs sit under the image rather than beside it, so their
# labels are kept small enough not to compete with it.
DETAIL_TAB_STYLE = """
QTabBar::tab {
    font-size: 10px;
    font-weight: normal;
    padding: 3px 10px;
    min-width: 0px;
}
"""


class ImageWorkspace(QtWidgets.QWidget):
    """
    Central image and profile workspace.

    Layout: per-slice viewer with slice slider and movable ROI box, and a
    splitter beneath the slider giving on to the ROI graphs and stack details.

    """

    # Profile choices, in the order they are stacked behind the selector.
    PROFILES = ("Z-axis (slicewise)", "Horizontal (column)", "Vertical (row)")

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Empty initial state.
        self._raw_stack: Optional[np.ndarray] = None
        self._current_stack: Optional[Stack] = None
        self._n_frames = 0
        self._current_frame = 0

        root = vbox(self, (4, 4, 4, 4), 4)

        self._body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._body_splitter.setChildrenCollapsible(False)
        root.addWidget(self._body_splitter, stretch=1)

        ################
        # IMAGE VIEWER #
        ################

        img_pane = QtWidgets.QWidget()
        img_row = hbox(img_pane, (0, 0, 0, 0), 0)
        raw_col = vbox()

        self._raw_lbl = label(
            "Ready", "font-weight: bold; font-size: 11px;", center=True
        )
        raw_col.addWidget(self._raw_lbl)

        self._raw_glw = pg.GraphicsLayoutWidget()
        self._raw_glw.setMinimumHeight(200)
        self._raw_glw.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self._raw_vb = self._raw_glw.addViewBox(lockAspect=True)
        self._raw_img = pg.ImageItem()
        self._raw_vb.addItem(self._raw_img)
        raw_col.addWidget(self._raw_glw, stretch=1)

        # Frame slider under the image view
        raw_col.addLayout(self._build_slider_row())
        img_row.addLayout(raw_col)

        # Roi box
        self._roi = pg.RectROI(
            pos=[60, 30],
            size=[25, 70],
            pen=pg.mkPen("y", width=2),
            hoverPen=pg.mkPen("y", width=3),
        )
        # Resize handles on all four edges
        for handle, centre in (
            ([1, 0.5], [0, 0.5]),
            ([0, 0.5], [1, 0.5]),
            ([0.5, 0], [0.5, 1]),
            ([0.5, 1], [0.5, 0]),
        ):
            self._roi.addScaleHandle(handle, centre)
        self._roi.sigRegionChanged.connect(self._on_roi_moved)
        self._raw_vb.addItem(self._roi)

        # ROI label
        self._roi_text = pg.TextItem("ROI", color="y", anchor=(0, 1))
        self._raw_vb.addItem(self._roi_text)

        # ROI stats info box
        self._roi_info = QtWidgets.QLabel(self._raw_glw.viewport())
        self._roi_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._roi_info.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._roi_info.setStyleSheet(
            "QLabel { background: rgba(20, 20, 20, 190); color: #eeeeee;"
            " border: 1px solid #888; border-radius: 4px; padding: 4px 6px;"
            " font-size: 10px; }"
        )
        self._roi_info.hide()
        self._raw_glw.viewport().installEventFilter(self)

        img_pane.setMinimumHeight(240)
        self._body_splitter.addWidget(img_pane)

        #############
        # TABS AREA #
        #############

        self._detail_tabs = QtWidgets.QTabWidget()
        self._detail_tabs.setStyleSheet(DETAIL_TAB_STYLE)
        self._detail_tabs.addTab(self._build_roi_graphs_tab(), "ROI Graphs")
        self._detail_tabs.addTab(self._build_stack_details_tab(), "Stack Details")
        self._detail_tabs.setMinimumHeight(160)
        self._body_splitter.addWidget(self._detail_tabs)

        self._body_splitter.setStretchFactor(0, 3)
        self._body_splitter.setStretchFactor(1, 2)
        self._body_splitter.setSizes([540, 360])

        self._update_details()

    ################
    # TAB BUILDERS #
    ################

    @staticmethod
    def _plot(left: str, bottom: str) -> pg.PlotWidget:
        """Return a PlotWidget with the given axis labels."""
        p = pg.PlotWidget()
        p.setLabel("left", left)
        p.setLabel("bottom", bottom)
        return p

    def _build_roi_graphs_tab(self) -> QtWidgets.QWidget:
        """
        Plot ROI profiles.
        """
        tab = QtWidgets.QWidget()
        outer = vbox(tab, (4, 4, 4, 4), 2)

        self._z_plot = self._plot("Intensity", "Frame")
        self._h_plot = self._plot("Intensity", "Column")
        self._vert_plot = self._plot("Row", "Intensity")

        self._plot_stack = QtWidgets.QStackedWidget()
        for plot in (self._z_plot, self._h_plot, self._vert_plot):
            plot.getAxis("left").setWidth(60)
            plot.setMinimumHeight(120)
            self._plot_stack.addWidget(plot)  # order matches PROFILES

        outer.addLayout(self._build_wavelength_row())
        outer.addWidget(self._plot_stack, stretch=1)
        return tab

    def _build_wavelength_row(self) -> QtWidgets.QHBoxLayout:
        """
        Build the profile selector and the experiment inputs (collimation distance/ delay)
        that the z-profile's wavelength axis needs.
        """
        row = hbox(spacing=4)

        self._profile_choice = QtWidgets.QComboBox()
        self._profile_choice.addItems(list(self.PROFILES))
        self._profile_choice.setToolTip("Which ROI profile to plot.")
        self._profile_choice.currentIndexChanged.connect(
            self._plot_stack.setCurrentIndex
        )
        row.addWidget(self._profile_choice)

        self._collimation_spin = dspin(0.0, 0.0, 1e9, 4)
        self._collimation_spin.setToolTip(
            "Flight-path length (m) times of flight are converted over. "
            "Zero leaves the profile in frames."
        )
        self._collimation_spin.setMaximumWidth(90)
        self._collimation_spin.valueChanged.connect(self._refresh_profiles)
        row.addWidget(label("collimation (m)", "font-size: 10px; color: #444;"))
        row.addWidget(self._collimation_spin)

        self._delay_spin = dspin(0.0, 0.0, 1e9, 6)
        self._delay_spin.setToolTip("Acquisition delay (s) added to the frame times.")
        self._delay_spin.setMaximumWidth(90)
        self._delay_spin.valueChanged.connect(self._refresh_profiles)
        row.addWidget(label("delay (s)", "font-size: 10px; color: #444;"))
        row.addWidget(self._delay_spin)

        self._z_axis_lbl = label("", "font-size: 10px; color: #888;")
        row.addWidget(self._z_axis_lbl)
        row.addStretch()
        return row

    def _z_axis(self, n_frames: int) -> tuple[np.ndarray, str, str]:
        """
        Return the z-profile's x values, its axis label, and a status note reporting what has been returned.
        Plots wavelengths where possible.

        Parameters
        ----------
        n_frames : int
            Number of points in the z-profile.

        Returns
        -------
        tuple[np.ndarray, str, str]
            (x values, bottom-axis label, note for the controls row).
        """
        frames = np.arange(n_frames)
        distance = float(self._collimation_spin.value())
        if distance <= 0:
            return frames, "Frame", "set a collimation distance to plot wavelengths"
        wavelengths = self._current_stack.stack_meta.get("wavelengths", None)
        if wavelengths is None:
            wavelengths =  self._current_stack.wavelengths(
            float(self._delay_spin.value()), distance
            )
        else:
            wavelengths = np.array(wavelengths)
        if wavelengths is None or wavelengths.size < n_frames:
            return frames, "Frame", "no frame times on this stack : plotting frames"
        return wavelengths[:n_frames], "Wavelength (Å)", ""

    def _z_value_label(self) -> str:
        """
        Return the z-profile's y-axis label for the previewed stack, e.g. transmission, intensity, cross-section.
        """
        stack = getattr(self, "_current_stack", None)
        if stack is None:
            return "Intensity"
        for step in reversed(stack.process_history()):
            axis_label = Z_AXIS_LABELS.get(step.get("process"))
            if axis_label:
                return axis_label
        return "Intensity"

    def _build_stack_details_tab(self) -> QtWidgets.QWidget:
        """Build the "Stack Details" tab."""
        tab = QtWidgets.QWidget()
        outer = vbox(tab, (8, 8, 8, 8), 6)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        body = QtWidgets.QWidget()
        form = vbox(body, (0, 0, 0, 0), 10)

        def _value_label(mono: bool = False) -> QtWidgets.QLabel:
            """Return a selectable, word-wrapped value label (optionally monospaced)."""
            style = ("font-family: monospace; " if mono else "") + "font-size: 11px;"
            lbl = label("(no stack loaded)", style, wrap=True)
            lbl.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            return lbl

        section_style = "font-weight: bold; font-size: 12px; color: #333;"
        self._detail_path = _value_label(mono=True)
        self._detail_pipeline = _value_label()
        self._detail_analysis = _value_label(mono=True)
        for title, value in (
            ("Stack path", self._detail_path),
            ("Processing pipeline", self._detail_pipeline),
            ("Analysis / compute results", self._detail_analysis),
        ):
            form.addWidget(label(title, section_style))
            form.addWidget(value)

        form.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return tab

    def _update_details(self) -> None:
        """Refresh the "Stack Details" tab using the currently previewed stack."""
        stack = getattr(self, "_current_stack", None)
        if stack is None:
            for lbl in (
                self._detail_path,
                self._detail_pipeline,
                self._detail_analysis,
            ):
                lbl.setText("(no stack loaded)")
            return
        path = getattr(stack, "path", None)
        self._detail_path.setText("(in-memory stack)" if path is None else str(path))
        self._detail_pipeline.setText(stack.process_history_string())
        self._detail_analysis.setText(stack.analysis_results_text())

    #################
    # ROI STATS BOX #
    #################

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Keep the ROI info box pinned to the top-right of the preview window."""
        if (
            obj is self._raw_glw.viewport()
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            self._reposition_roi_info()
        return super().eventFilter(obj, event)

    def _reposition_roi_info(self) -> None:
        """Move the ROI info box back to the top-right corner of the preview viewport."""
        info = getattr(self, "_roi_info", None)
        if info is None:
            return
        margin = 8
        vp = self._raw_glw.viewport()
        info.move(max(margin, vp.width() - info.width() - margin), margin)

    def _update_roi_info(self, stats: Optional[dict]) -> None:
        """Update the top-right ROI info box from a compute_roi_stats result."""
        info = getattr(self, "_roi_info", None)
        if info is None:
            return
        if not stats:
            info.hide()
            return

        rows = "<br>".join(
            f"{key}: {stats.get(key, float('nan')):.4g}"
            for key in ("mean", "median", "std", "sem", "min", "max")
        )
        text = f"<b>ROI stats</b><br>{rows}<br>n: {stats.get('valid_pixels', 0)}"
        if text == info.text() and not info.isHidden():
            return  # nothing to repaint
        info.setText(text)
        info.adjustSize()
        if info.isHidden():
            # Raising is only needed the first time it appears; doing it on
            # every drag event is what makes the trail worse.
            info.show()
            info.raise_()
        self._reposition_roi_info()

    ###########
    # SLIDERS #
    ###########

    def _build_slider_row(self) -> QtWidgets.QHBoxLayout:
        """Create the labelled frame-slider row under the image view."""
        row = hbox()

        lbl = label("Frame\nslider", "font-size: 10px;", center=True)
        lbl.setFixedWidth(44)
        row.addWidget(lbl)

        self._raw_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._raw_slider.setRange(0, 99)  # maximum updated after data is loaded
        self._raw_slider.setValue(0)
        self._raw_slider.valueChanged.connect(self._show_frame)
        row.addWidget(self._raw_slider)

        self._raw_frame_lbl = label("0", center=True)
        self._raw_frame_lbl.setFixedWidth(30)
        row.addWidget(self._raw_frame_lbl)
        return row

    ##############
    # FRAME VIEW #
    ##############

    def _show_frame(self, idx: int) -> None:
        """
        Update the image view and all profile plots for frame index.
        Called by the slider and by load_stack; does nothing while the
        previewer is empty.
        """
        if self._raw_stack is None:
            return

        self._current_frame = idx
        self._raw_img.setImage(self._raw_stack[idx])

        # Update the ROI text label position
        rp = self._roi.pos()
        self._roi_text.setPos(rp.x(), rp.y())

        self._raw_frame_lbl.setText(str(idx))
        self._refresh_profiles()

    def _refresh_profiles(self) -> None:
        """
        Extract ROI data from the current frame and update the horizontal,
        vertical and z-axis (all-frames) mean-profile plots plus the stats box.
        """
        if self._raw_stack is None:
            return  # previewer is empty (no stack loaded)

        rp, rs = self._roi.pos(), self._roi.size()
        h, w = self._raw_stack.shape[1], self._raw_stack.shape[2]

        # Clamp ROI to image bounds
        c0 = max(0, int(rp.x()))
        r0 = max(0, int(rp.y()))
        c1 = min(w, c0 + max(1, int(rs.x())))
        r1 = min(h, r0 + max(1, int(rs.y())))

        if c1 <= c0 or r1 <= r0:
            self._update_roi_info(None)  # non-functional ROI
            return  # nothing to show

        frame_crop = self._raw_stack[self._current_frame, r0:r1, c0:c1]

        # Horizontal profile: row-mean of the ROI slice for this frame
        h_profile = frame_crop.mean(axis=0)
        self._h_plot.clear()
        self._h_plot.plot(
            np.arange(c0, c0 + len(h_profile)),
            h_profile,
            pen=pg.mkPen("#4fc3f7", width=1.5),
        )

        # Vertical profile: column-mean of the ROI slice for this frame
        v_profile = frame_crop.mean(axis=1)
        self._vert_plot.clear()
        self._vert_plot.plot(
            v_profile,
            np.arange(r0, r0 + len(v_profile)),
            pen=pg.mkPen("#ce93d8", width=1.5),
        )

        z_profile = self._raw_stack[:, r0:r1, c0:c1].mean(axis=(1, 2))
        x_values, x_label, note = self._z_axis(len(z_profile))
        self._z_plot.setLabel("bottom", x_label)
        self._z_plot.setLabel("left", self._z_value_label())
        self._z_axis_lbl.setText(note)
        self._z_plot.clear()
        self._z_plot.plot(x_values, z_profile, pen=pg.mkPen("#a5d6a7", width=1.5))

        # Line marking the current frame position, wherever it lands on the axis
        if len(x_values):
            frame = min(self._current_frame, len(x_values) - 1)
            position = float(x_values[frame])
            vline = pg.InfiniteLine(
                pos=position,
                angle=90,
                pen=pg.mkPen("#ff5555", width=1, style=QtCore.Qt.PenStyle.DashLine),
                label=(
                    f"λ={position:.3f} Å"
                    if x_label.startswith("Wavelength")
                    else f"f={self._current_frame}"
                ),
                labelOpts={"color": "#ff5555", "fill": "#33333388"},
            )
            self._z_plot.addItem(vline, ignoreBounds=True)
        try:
            stats = compute_roi_stats(
                self._raw_stack[self._current_frame], (c0, r0, (c1 - c0), (r1 - r0))
            )
        except Exception:
            stats = None
        self._update_roi_info(stats)

    def _on_roi_moved(self) -> None:
        """Called by pyqtgraph whenever the ROI is dragged or resized."""
        rp = self._roi.pos()
        self._roi_text.setPos(rp.x(), rp.y())
        self._refresh_profiles()

    def current_roi_xywh(self) -> tuple[int, int, int, int]:
        """
        Return the current ROI box as (x, y, w, h) in image pixel
        coordinates (x = column, y = row), with width and height at least 1.
        """
        rp, rs = self._roi.pos(), self._roi.size()
        x = int(round(rp.x()))
        y = int(round(rp.y()))
        w = max(1, int(round(rs.x())))
        h = max(1, int(round(rs.y())))
        return x, y, w, h

    def set_roi_xywh(self, x: int, y: int, w: int, h: int) -> None:
        """
        Position and size the ROI box (image pixel coordinates) and refresh the
        profile plots : used to re-apply a saved ROI to the previewer.
        """
        self._roi.blockSignals(True)
        self._roi.setPos([float(x), float(y)])
        self._roi.setSize([float(max(1, w)), float(max(1, h))])
        self._roi.blockSignals(False)
        self._roi_text.setPos(float(x), float(y))
        self._refresh_profiles()

    def load_stack(self, stack: Stack, name: str = "") -> None:
        """
        Display a stack in the previewer: the frame slider
        range, ROI position and all profile plots are updated to match the new
        data, and the "Stack Details" tab is refreshed.
        """
        data = np.asarray(stack.data, dtype=np.float32)
        if data.ndim == 2:  # single frame → 1-frame stack
            data = data[None, :, :]

        self._raw_stack = data
        self._current_stack = stack
        self._n_frames = data.shape[0]
        self._raw_lbl.setText(f"{name or 'stack'}")

        self._update_details()

        # Re-centre the ROI so it always lands inside the new image bounds
        h, w = data.shape[1], data.shape[2]
        self._roi.blockSignals(True)
        self._roi.setPos([w * 0.30, h * 0.30])
        self._roi.setSize([max(4.0, w * 0.30), max(4.0, h * 0.30)])
        self._roi.blockSignals(False)

        self._raw_slider.blockSignals(True)
        self._raw_slider.setMaximum(max(0, self._n_frames - 1))
        self._raw_slider.setValue(0)
        self._raw_slider.blockSignals(False)

        self._show_frame(0)
        self._raw_vb.autoRange()
