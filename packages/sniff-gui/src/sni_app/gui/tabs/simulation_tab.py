"""
Simulation tab: AFGA predicts cross sections.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from molmass import Formula
from PyQt6 import QtCore, QtWidgets

from sni_app.core.process.afga import (
    SPEC_BUILDING_BLOCKS,
    WAVELENGTH_MAX_A,
    WAVELENGTH_MIN_A,
    process_compound,
)
from sni_app.gui.panels.logging import StatusPanel
from sni_app.gui.shared import (
    BTN_STYLE_RED,
    JobRunnerMixin,
    confirm,
    dspin,
    eliding_label,
    flat_btn,
    hbox,
    label,
    popup,
    run_btn,
    vbox,
    vscroll,
)

_PLOT_COLORS: Tuple[str, ...] = (
    "#4fc3f7",
    "#ce93d8",
    "#81c784",
    "#ffb74d",
    "#e57373",
    "#64b5f6",
    "#f06292",
    "#a1887f",
    "#9ccc65",
    "#4db6ac",
)


# per molecule = per atom * (number of atoms in molecule)
SCALE_BY_ATOM = "By atom"
SCALE_BY_MOLECULE = "By molecule"
SCALING_MODES: Tuple[str, ...] = (SCALE_BY_MOLECULE, SCALE_BY_ATOM)

# one graph per plot type
GRAPH_TITLES = {
    SCALE_BY_ATOM: "Microscopic Neutron Cross Sections, per atom",
    SCALE_BY_MOLECULE: "Microscopic Neutron Cross Sections, per molecule",
}
GRAPH_Y_LABELS = {
    SCALE_BY_ATOM: "Microscopic Neutron Cross Section (barns per atom)",
    SCALE_BY_MOLECULE: "Microscopic Neutron Cross Section (barns p.f.u)",
}

SPEC_BTN_MIN_SIZE = (56, 26) # for building block buttons


def scaling_factor_for(mode: str, formula: str) -> float:
    """
    Return the cross-section scaling factor for a plotting mode.

    Parameters
    ----------
    mode : str
        Either 'By atom' or 'By molecule'.
    formula : str
        Chemical formula

    Returns
    -------
    float
        1.0 when plotting by atom, else the number of atoms in the formula.

    Raises
    ------
    ValueError
        If the formula cannot be read as a chemical formula.
    """
    if mode != SCALE_BY_MOLECULE:
        return 1.0
    try:
        atoms = int(Formula(formula).atoms)
    except Exception as exc:  # molmass raises a range of parse errors
        raise ValueError(f"Cannot count the atoms in formula '{formula}': {exc}")
    if atoms <= 0:
        raise ValueError(f"Formula '{formula}' holds no atoms.")
    return float(atoms)


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex' to an (r, g, b) tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@dataclass
class PlotRecord:
    """One curve on the graph and its parameters."""

    id: int
    name: str
    params: Dict[str, object]
    x: np.ndarray
    y: np.ndarray
    color: Tuple[int, int, int]
    visible: bool = True

    def curve_summary(self) -> str:
        """Return a summary of the parameters for row."""
        p = self.params
        return (
            f"{self.name}\n"
            f"formula: {p['formula']}\n"
            f"spec: {p['spec']}\n"
            f"density: {p['density']} g/cm³   "
            f"plotted: {p['scaling_mode'].lower()} (x{p['scaling_factor']:g})\n"
            f"Debye T: {p['temperature']} K"
        )


class SpecBuilder(QtWidgets.QWidget):
    """
    Read-only spec constructed from building-block buttons, which are given minimum size.
    """

    spec_changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        blocks: Tuple[str, ...],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tokens: List[str] = []  # click sequence, e.g. ["CH3", "CH3", "CHali"]

        lay = vbox(self, (0, 0, 0, 0), 4)

        self._field = QtWidgets.QLineEdit()
        self._field.setReadOnly(True)
        self._field.setPlaceholderText("Build Spec...")
        self._field.setToolTip("HFG spec: build from the buttons below")
        lay.addWidget(self._field)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for i, block in enumerate(blocks):
            btn = QtWidgets.QPushButton(block)
            btn.setToolTip(f"Add one {block} group")
            btn.setMinimumSize(*SPEC_BTN_MIN_SIZE)
            btn.clicked.connect(lambda _=False, b=block: self._add(b))
            grid.addWidget(btn, i // 3, i % 3)
        lay.addLayout(grid)

        edit_row = hbox(margins=(0, 0, 0, 0), spacing=4)
        back_btn = QtWidgets.QPushButton("Remove last")
        back_btn.setToolTip("Remove the most recently added group")
        back_btn.setMinimumSize(*SPEC_BTN_MIN_SIZE)
        back_btn.clicked.connect(self._remove_last)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setToolTip("Clear the spec")
        clear_btn.setMinimumSize(*SPEC_BTN_MIN_SIZE)
        clear_btn.clicked.connect(self.clear)
        edit_row.addWidget(back_btn)
        edit_row.addWidget(clear_btn)
        edit_row.addStretch()
        lay.addLayout(edit_row)

    def spec(self) -> str:
        """Return the current spec string."""
        return self._field.text()

    def clear(self) -> None:
        """Empty the spec."""
        self._tokens = []
        self._refresh()

    def _add(self, block: str) -> None:
        self._tokens.append(block)
        self._refresh()

    def _remove_last(self) -> None:
        if self._tokens:
            self._tokens.pop()
            self._refresh()

    def _refresh(self) -> None:
        self._field.setText(self._render(self._tokens))
        self.spec_changed.emit(self._field.text())

    @staticmethod
    def _render(tokens: List[str]) -> str:
        """Aggregate tokens into block groups, preserving first-appearance order."""
        order: List[str] = []
        counts: Dict[str, int] = {}
        for token in tokens:
            if token not in counts:
                order.append(token)
                counts[token] = 0
            counts[token] += 1
        return "+".join(f"{counts[b]}x{b}" if counts[b] > 1 else b for b in order)


class PlotRow(QtWidgets.QWidget):
    """
    One row in the Plots box: colour swatch, name, rename (✎), show/hide and
    delete (x).  Changes are emitted to the tab, which owns the records.
    """

    rename_requested = QtCore.pyqtSignal(int, str)
    remove_requested = QtCore.pyqtSignal(int)
    visibility_toggled = QtCore.pyqtSignal(int, bool)

    def __init__(
        self, record: PlotRecord, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._id = record.id
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "PlotRow { border-radius: 4px; }PlotRow:hover { background: #3a3a3a; }"
        )

        row = hbox(self, (4, 2, 4, 2), 4)

        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(14, 14)
        r, g, b = record.color
        swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border-radius: 3px;")
        row.addWidget(swatch)

        self._name_lbl = eliding_label(
            record.name, QtCore.Qt.TextElideMode.ElideRight, "font-weight: bold;"
        )
        self._name_lbl.setToolTip(record.curve_summary())
        row.addWidget(self._name_lbl, stretch=1)

        rename_btn = flat_btn(
            "✎",
            "Rename plot",
            "QPushButton { border: none; }QPushButton:hover { color: #4fc3f7; }",
        )
        rename_btn.clicked.connect(self._rename)
        row.addWidget(rename_btn)

        self._vis = QtWidgets.QCheckBox()
        self._vis.setChecked(record.visible)
        self._vis.setToolTip("Show or hide this plot")
        self._vis.toggled.connect(
            lambda checked: self.visibility_toggled.emit(self._id, checked)
        )
        row.addWidget(self._vis)

        remove_btn = flat_btn(
            "x",
            "Delete this plot",
            "QPushButton { border: none; color: #cc6666; font-weight: bold; }"
            "QPushButton:hover { color: #ff5555; }",
        )
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._id))
        row.addWidget(remove_btn)

    def _rename(self) -> None:
        """Prompt for a new display name and request it from the tab."""
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Plot", "New name:", text=self._name_lbl.text()
        )
        if ok and new.strip():
            self.rename_requested.emit(self._id, new.strip())


class PlotListBox(QtWidgets.QGroupBox):
    """Scrollable list of PlotRows with a 'Clear Plots' footer."""

    rename_requested = QtCore.pyqtSignal(int, str)
    remove_requested = QtCore.pyqtSignal(int)
    visibility_toggled = QtCore.pyqtSignal(int, bool)
    clear_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Plots", parent)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")
        self._rows: List[PlotRow] = []

        lay = vbox(self, (4, 4, 4, 4))

        body = QtWidgets.QWidget()
        self._rows_layout = vbox(body, (2, 2, 2, 2), 2)
        self._empty_lbl = label(
            "No plots yet.", "color: #888; font-style: italic; font-size: 11px;"
        )
        self._rows_layout.addWidget(self._empty_lbl)
        self._rows_layout.addStretch()
        lay.addWidget(vscroll(body))

        clear_row = hbox(margins=(0, 0, 0, 0))
        clear_row.addStretch()
        self._clear_btn = QtWidgets.QPushButton("Clear Plots")
        self._clear_btn.setStyleSheet(BTN_STYLE_RED)
        self._clear_btn.setToolTip("Remove all plots")
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        clear_row.addWidget(self._clear_btn)
        lay.addLayout(clear_row)

    def set_records(self, records: List[PlotRecord]) -> None:
        """Rebuild every row from records."""
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []

        for record in records:
            row = PlotRow(record)
            row.rename_requested.connect(self.rename_requested)
            row.remove_requested.connect(self.remove_requested)
            row.visibility_toggled.connect(self.visibility_toggled)
            self._rows.append(row)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)

        self._empty_lbl.setVisible(not records)
        self._clear_btn.setEnabled(bool(records))


class SimulationTab(JobRunnerMixin, QtWidgets.QWidget):
    """
    AFGA tab, used to predict cross-sections for experiments.
    """

    def __init__(
        self,
        status: Optional[StatusPanel] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._status = status
        self._records: List[PlotRecord] = []
        self._next_id = 1
        self._color_idx = 0

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1000])
        hbox(self, (0, 0, 0, 0), 0).addWidget(splitter)

        self._refresh()
        self._sync_add_enabled()

    ###################
    # WIDGET BUILDERS #
    ###################

    def _build_left(self) -> QtWidgets.QWidget:
        """Build the input form, Add button and Plots box (left column)."""
        col = QtWidgets.QWidget()
        col.setMinimumWidth(320)
        col.setMaximumWidth(440)
        lay = vbox(col, (6, 6, 6, 6), 6)

        form_box = QtWidgets.QGroupBox("Compound")
        form_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        form = QtWidgets.QFormLayout(form_box)

        self._name_edit = QtWidgets.QLineEdit("")
        self._formula_edit = QtWidgets.QLineEdit("")
        self._formula_edit.setToolTip("Chemical formula")
        self._spec = SpecBuilder(SPEC_BUILDING_BLOCKS)
        self._density = dspin(0.932, 0.0, 50.0, decimals=4, step=0.01)
        self._scaling_mode = QtWidgets.QComboBox()
        self._scaling_mode.addItems(list(SCALING_MODES))
        self._scaling_mode.setToolTip(
            "Plot the cross section per atom, or per molecule (scaled by "
            "the number of atoms in the formula)."
        )
        self._temperature = dspin(283.15, 1.0, 2000.0, decimals=2, step=1.0)

        self._wl_min = dspin(WAVELENGTH_MIN_A, 0.01, 1e3, decimals=3, step=0.1)
        self._wl_max = dspin(WAVELENGTH_MAX_A, 0.01, 1e3, decimals=3, step=0.1)
        for spin, end in ((self._wl_min, "Shortest"), (self._wl_max, "Longest")):
            spin.setToolTip(f"{end} wavelength the cross section is computed over.")

        form.addRow("Name", self._name_edit)
        form.addRow("Formula", self._formula_edit)
        form.addRow("Spec", self._spec)
        form.addRow("Density (g/cm³)", self._density)
        form.addRow("Plot by", self._scaling_mode)
        form.addRow("Debye temp. (K)", self._temperature)
        form.addRow("λ min (Å)", self._wl_min)
        form.addRow("λ max (Å)", self._wl_max)
        lay.addWidget(form_box)

        self._add_btn = run_btn(self._add_plot, "Add Plot")
        self._add_btn.setToolTip("Compute the cross section and add it to the graph")
        lay.addWidget(self._add_btn)

        self._plot_box = PlotListBox()
        self._plot_box.rename_requested.connect(self._rename_plot)
        self._plot_box.remove_requested.connect(self._remove_plot)
        self._plot_box.visibility_toggled.connect(self._set_visibility)
        self._plot_box.clear_requested.connect(self._clear_plots)
        lay.addWidget(self._plot_box, stretch=1)

        self._name_edit.textChanged.connect(self._sync_add_enabled)
        self._formula_edit.textChanged.connect(self._sync_add_enabled)
        self._spec.spec_changed.connect(self._sync_add_enabled)
        return col

    def _build_center(self) -> QtWidgets.QWidget:
        """
        Build the graphs (centre): one per scaling mode, behind a selector.
        """
        w = QtWidgets.QWidget()
        lay = vbox(w, (0, 0, 0, 0), 4)

        self._plots: Dict[str, pg.PlotWidget] = {}
        self._legends: Dict[str, object] = {}
        self._graph_stack = QtWidgets.QStackedWidget()
        for mode in SCALING_MODES:
            plot = pg.PlotWidget()
            plot.setLabel("bottom", "Wavelength (Å)")
            plot.setLabel("left", GRAPH_Y_LABELS[mode])
            plot.setTitle(GRAPH_TITLES[mode])
            plot.showGrid(x=True, y=True, alpha=0.2)
            self._legends[mode] = plot.addLegend(offset=(-10, 10))
            self._plots[mode] = plot
            self._graph_stack.addWidget(plot)  # order matches SCALING_MODES

        view_row = hbox(margins=(4, 4, 4, 0), spacing=6)
        view_row.addWidget(label("Graph", "font-size: 11px; color: #aaa;"))
        self._graph_choice = QtWidgets.QComboBox()
        self._graph_choice.addItems(list(SCALING_MODES))
        self._graph_choice.setToolTip("Which of the two graphs to show.")
        self._graph_choice.currentIndexChanged.connect(
            self._graph_stack.setCurrentIndex
        )
        self._graph_choice.currentIndexChanged.connect(lambda _idx: self._redraw())
        view_row.addWidget(self._graph_choice)
        self._graph_count_lbl = label("", "font-size: 10px; color: #888;")
        view_row.addWidget(self._graph_count_lbl)
        view_row.addStretch()
        lay.addLayout(view_row)

        lay.addWidget(self._graph_stack, stretch=1)
        return w

    def _show_graph(self, mode: str) -> None:
        """Bring the graph for a scaling mode to the front."""
        index = self._graph_choice.findText(mode)
        if index >= 0:
            self._graph_choice.setCurrentIndex(index)

    #################
    # COMPUTE / ADD #
    #################

    def _add_plot(self) -> None:
        """Compute the current compound off-thread and add it as a plot."""
        if self.job_running():
            return
        try:
            params = self._current_params()
        except ValueError as exc:  # unreadable formula in by-molecule mode
            popup(self, "Invalid Formula", str(exc))
            return
        if not params["formula"] or not params["spec"]:
            popup(self, "Missing Input", "Enter a formula and build a spec first.")
            return
        if params["wavelength_max"] <= params["wavelength_min"]:
            popup(self, "Invalid Range", "Max wavelength must be greater than min wavelength.")
            return
        name = params["name"] or "compound"
        # scaling_mode is a record of how scaling_factor was chosen, not an
        # argument the computation itself takes.
        compute_args = {k: v for k, v in params.items() if k != "scaling_mode"}

        def job(_report):
            return process_compound(**compute_args)

        self.run_job(
            job,
            f"Compute '{name}'",
            lambda result: self._on_computed(name, params, result),
        )

    def _current_params(self) -> Dict[str, object]:
        """
        Snapshot the form into process_compound()'s keyword arguments,
        plus the scaling_mode the factor was derived from (recorded on the
        curve, and stripped before the computation is called).
        """
        formula = self._formula_edit.text().strip()
        mode = self._scaling_mode.currentText()
        return {
            "name": self._name_edit.text().strip(),
            "formula": formula,
            "spec": self._spec.spec().strip(),
            "density": float(self._density.value()),
            "scaling_mode": mode,
            "scaling_factor": scaling_factor_for(mode, formula) if formula else 1.0,
            "temperature": float(self._temperature.value()),
            "wavelength_min": float(self._wl_min.value()),
            "wavelength_max": float(self._wl_max.value()),
        }

    def _on_computed(self, name: str, params: Dict[str, object], result: Tuple) -> None:
        """A compute job finished; record the curve and redraw."""
        x, y = result
        color = _hex_to_rgb(_PLOT_COLORS[self._color_idx % len(_PLOT_COLORS)])
        self._color_idx += 1
        record = PlotRecord(
            id=self._next_id,
            name=self._unique_name(name),
            params=params,
            x=np.asarray(x),
            y=np.asarray(y),
            color=color,
        )
        self._next_id += 1
        self._records.append(record)
        self._show_graph(str(params.get("scaling_mode", SCALE_BY_ATOM)))
        self._refresh()
        self._log(f"Added plot '{record.name}'.")

    def _set_job_busy(self, busy: bool) -> None:
        """Gate the Add button while a compute job runs (JobRunnerMixin hook)."""
        self._add_btn.setEnabled(not busy and self._inputs_valid())

    ###################
    # PLOT LIST EDITS #
    ###################

    def _find(self, plot_id: int) -> Optional[PlotRecord]:
        return next((r for r in self._records if r.id == plot_id), None)

    def _rename_plot(self, plot_id: int, new_name: str) -> None:
        record = self._find(plot_id)
        if record is None:
            return
        record.name = self._unique_name(new_name, exclude=plot_id)
        self._refresh()
        self._log(f"Renamed plot to '{record.name}'.")

    def _remove_plot(self, plot_id: int) -> None:
        record = self._find(plot_id)
        if record is None:
            return
        if confirm(self, "Delete Plot", f"Delete plot '{record.name}'?"):
            self._records = [r for r in self._records if r.id != plot_id]
            self._refresh()
            self._log(f"Deleted plot '{record.name}'.")

    def _set_visibility(self, plot_id: int, visible: bool) -> None:
        record = self._find(plot_id)
        if record is not None:
            record.visible = visible
            self._redraw()

    def _clear_plots(self) -> None:
        if self._records and confirm(self, "Clear Plots", "Remove all plots?"):
            self._records = []
            self._refresh()
            self._log("Cleared all plots.")

    def _unique_name(self, base: str, exclude: Optional[int] = None) -> str:
        """Return base, suffixed '(n)' if another record already uses it."""
        base = base or "compound"
        taken = {r.name for r in self._records if r.id != exclude}
        if base not in taken:
            return base
        i = 2
        while f"{base} ({i})" in taken:
            i += 1
        return f"{base} ({i})"

    ###########
    # DRAWING #
    ###########

    def _refresh(self) -> None:
        """Sync the Plots box and the graph to the current records."""
        self._plot_box.set_records(self._records)
        self._redraw()

    def _redraw(self) -> None:
        """
        Rebuild both graphs from the records (keeps them consistent with the
        box).  Each record is drawn on the graph for the mode it was computed
        in; anything computed before the modes existed is treated as per atom.
        """
        for mode in SCALING_MODES:
            self._plots[mode].clear()
            legend = self._legends.get(mode)
            if legend is not None:
                legend.clear()

        drawn = {mode: 0 for mode in SCALING_MODES}
        for record in self._records:
            mode = str(record.params.get("scaling_mode", SCALE_BY_ATOM))
            plot = self._plots.get(mode)
            if plot is None or not record.visible:
                continue
            plot.plot(
                record.x,
                record.y,
                pen=pg.mkPen(record.color, width=1.5),
                name=record.name,
            )
            drawn[mode] += 1

        other = next(m for m in SCALING_MODES if m != self._graph_choice.currentText())
        self._graph_count_lbl.setText(
            f"({drawn[other]} plot(s) on '{other}')" if drawn[other] else ""
        )

    ##########
    # GATING #
    ##########
    def _inputs_valid(self) -> bool:
        """Return whether formula and spec are both present."""
        return bool(self._formula_edit.text().strip() and self._spec.spec().strip())

    def _sync_add_enabled(self, *_) -> None:
        """Enable the Add button only with valid inputs and no job in flight."""
        self._add_btn.setEnabled(self._inputs_valid() and not self.job_running())
