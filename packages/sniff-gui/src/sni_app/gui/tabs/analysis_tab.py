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
SNIFF: analysis / compute panel.
"""

from typing import Callable, Dict, List, Optional

from PyQt6 import QtCore, QtWidgets
from sni_app.gui.panels.logging import StatusPanel
from sni_app.gui.shared import (
    PANEL_HEADER_STYLE,
    JobRunnerMixin,
    dspin,
    label,
    popup,
    run_btn,
    spin,
    vbox,
    vscroll,
)

from sni_app.core.process.roi_processes import (
    atten_coefficient,
    h_cross_section,
    relative_attenuation,
    sum_of_logs_relative_attenuation,
    t_cross_section,
)
FIELD_MIN_WIDTH = 84


def parse_float_list(text: str, caption: str) -> List[float]:
    """
    Read a comma-separated list of numbers out of a text field.

    Parameters
    ----------
    text : str
        Field contents, e.g. "1.321, 1.2".
    caption : str
        The field's label, used in the error message.

    Returns
    -------
    List[float]

    Raises
    ------
    ValueError
        If the field is empty or holds anything that is not a number.
    """
    values = parse_str_list(text, caption)
    try:
        return [float(v) for v in values]
    except ValueError:
        raise ValueError(f"{caption} must be a comma-separated list of numbers.")


def parse_str_list(text: str, caption: str) -> List[str]:
    """
    Read a comma-separated list of words out of a text field.

    Parameters
    ----------
    text : str
        Field contents, e.g. "C3H4O3, C4H6O3".
    caption : str
        The field's label, used in the error message.

    Returns
    -------
    List[str]

    Raises
    ------
    ValueError
        If the field holds nothing.
    """
    values = [part.strip() for part in text.replace(";", ",").split(",")]
    values = [part for part in values if part]
    if not values:
        raise ValueError(f"{caption} must not be empty.")
    return values


class ComputePanel(JobRunnerMixin, QtWidgets.QWidget):
    """
    Analysis / compute function runner.

    Signals
    -------
    stack_produced : a computation produced a stack (passes name, Stack)
    """

    stack_produced = QtCore.pyqtSignal(str, object)  # name, Stack

    RELATIVE_ATTENUATION = "Relative Attenuation"
    ATTENUATION_COEFFICIENT = "Attenuation Coefficient"
    TOTAL_MICRO_CROSS_SECTION = "Total Microscopic Cross Section"
    HYDROGEN_CROSS_SECTION = "Hydrogen Cross Section"

    def __init__(
        self,
        status: Optional["StatusPanel"] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._status = status
        self._stacks: List[tuple] = []  # [(name, Stack)]
        self._stack_combos: List[QtWidgets.QComboBox] = []
        self._widgets: dict = {}

        self.setMinimumWidth(320)
        self.setMaximumWidth(440)

        outer = vbox(self, (0, 0, 0, 0), 0)
        outer.addWidget(label("Analysis / Compute", PANEL_HEADER_STYLE))

        body = QtWidgets.QWidget()
        vb = vbox(body, (8, 8, 8, 8), 8)

        # Function drop-down
        self._combo = self._fit_field(QtWidgets.QComboBox())
        self._combo.addItems(list(self._BUILDERS))
        self._combo.setToolTip("Choose the computation to run")
        self._combo.currentTextChanged.connect(lambda _name: self._rebuild_params())
        vb.addWidget(self._combo)

        # Parameter panel
        self._panel = QtWidgets.QWidget()
        self._panel_layout = vbox(self._panel, (2, 2, 2, 2), 10)
        vb.addWidget(self._panel)

        self._run_btn = run_btn(self._on_run)
        vb.addWidget(self._run_btn)
        vb.addStretch()  # keep the functions at the top of the column
        outer.addWidget(vscroll(body), stretch=1)

        self._rebuild_params()

    def current_function(self) -> str:
        """Return the function currently selected in the drop-down."""
        return self._combo.currentText()

    def set_function(self, name: str) -> None:
        """Select a function in the drop-down."""
        index = self._combo.findText(name)
        if index >= 0:
            self._combo.setCurrentIndex(index)

    def set_all_stacks(self, pairs: List[tuple]) -> None:
        """Update every stack-selection combo from list of (name, Stack) pairs."""
        self._stacks = list(pairs)
        for combo in self._stack_combos:
            self._refill_combo(combo, self._stacks)
        self._sync_run_enabled()

    ###################
    # PARAMETER PANEL #
    ###################

    def _rebuild_params(self) -> None:
        """Rebuild the parameter panel by dispatching to the function's builder."""
        self._clear_layout(self._panel_layout)
        self._widgets = {}
        self._stack_combos = []  # the old function's combos have just gone
        self._BUILDERS[self.current_function()](self)
        self._sync_run_enabled()

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        """Recursively remove and delete every widget/sub-layout from layout."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _form(self) -> QtWidgets.QFormLayout:
        """Add a form to the parameter panel and return it for the builder to fill."""
        container = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self._panel_layout.addWidget(container)
        return form

    @staticmethod
    def _caption(text: str) -> QtWidgets.QLabel:
        """
        Return a wrapped row caption.
        """
        return label(text, "font-size: 11px; color: #444;", wrap=True)

    @staticmethod
    def _fit_field(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """
        Fit parameter field shrink to column width.
        """
        widget.setMinimumWidth(FIELD_MIN_WIDTH)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            widget.sizePolicy().verticalPolicy(),
        )
        if isinstance(widget, QtWidgets.QComboBox):
            widget.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            widget.setMinimumContentsLength(8)
        return widget

    def _note(self, text: str) -> None:
        """Add an italic, word-wrapped note line to the parameter panel."""
        self._panel_layout.addWidget(
            label(text, "color: #888; font-style: italic; font-size: 11px;", wrap=True)
        )

    def _stack_row(
        self, form: QtWidgets.QFormLayout, role: str, caption: str, tooltip: str = ""
    ) -> QtWidgets.QComboBox:
        combo = self._fit_field(QtWidgets.QComboBox())
        if tooltip:
            combo.setToolTip(tooltip)
        self._refill_combo(combo, self._stacks)
        self._stack_combos.append(combo)
        self._widgets[role] = combo
        form.addRow(self._caption(caption), combo)
        return combo

    def _row(
        self, form: QtWidgets.QFormLayout, role: str, caption: str, widget
    ) -> QtWidgets.QWidget:
        """Add a parameter row, storing widget in the panel's widget map."""
        self._fit_field(widget)
        self._widgets[role] = widget
        form.addRow(self._caption(caption), widget)
        return widget

    @staticmethod
    def _refill_combo(combo: QtWidgets.QComboBox, pairs: List[tuple]) -> None:
        """Repopulate a stack combo and selection parameters from a list of (name, Stack)"""
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for name, stack in pairs:
            combo.addItem(name, stack)
        idx = combo.findText(prev)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _thickness_row(self, form: QtWidgets.QFormLayout) -> None:
        """Add the sample thickness row every coefficient computation takes."""
        self._row(
            form, "d_cm", "d (cm)", dspin(1.0, 1e-6, 1e6, decimals=6, step=0.1)
        ).setToolTip("Sample thickness, in centimetres.")

    ####################
    # FUNCTION BUILDERS #
    ####################

    def _build_relative_attenuation(self) -> None:
        """
        Relative attenuation on a single stack.

        Method drop-down chooses between Standard and Sum-of-Logs methods. Bin
        factor and pre-filter options apply to Sum-of-Logs only.
        """
        form = self._form()
        self._stack_row(
            form, "stack", "Stack", "Stack to compute relative attenuation on"
        )

        method = self._row(form, "method", "Method", QtWidgets.QComboBox())
        method.addItems(["Standard", "Sum-of-Logs"])

        self._row(form, "sw0", "Short band start", spin(0, 0, 1_000_000))
        self._row(form, "sw1", "Short band stop", spin(1, 0, 1_000_000))
        self._row(form, "lw0", "Long band start", spin(0, 0, 1_000_000))
        self._row(form, "lw1", "Long band stop", spin(1, 0, 1_000_000))
        self._row(
            form, "eps", "EPS", dspin(1e-6, 0.0, 1e9, decimals=8, step=1e-6)
        ).setToolTip("Minimum absolute short-band log value for a valid pixel.")

        # Sum-of-Logs-only options (enabled only for that method).
        self._row(form, "bin_factor", "Bin Factor", spin(1, 1, 1000))
        self._row(form, "filter_enabled", "Pre-filter", QtWidgets.QCheckBox())
        filter_mode = self._row(
            form, "filter_mode", "Filter Type", QtWidgets.QComboBox()
        )
        filter_mode.addItems(["None", "Median", "Gaussian"])
        self._row(form, "median_size", "Median Size", spin(3, 1, 99))
        self._row(
            form, "gauss_sigma", "σ", dspin(1.0, 0.0, 100.0, decimals=3, step=0.1)
        )

        method.currentTextChanged.connect(self._update_ra_method)
        self._update_ra_method(method.currentText())

    def _update_ra_method(self, method: str) -> None:
        """Grey out the Sum-of-Logs-only options unless that method is chosen."""
        for role in (
            "bin_factor",
            "filter_enabled",
            "filter_mode",
            "median_size",
            "gauss_sigma",
        ):
            widget = self._widgets.get(role)
            if widget is not None:
                widget.setEnabled(method == "Sum-of-Logs")

    def _build_atten_coefficient(self) -> None:
        """Attenuation coefficient of a sample against an empty sample holder."""
        form = self._form()
        self._stack_row(form, "stack", "Stack", "Stack for coefficient calculation")
        self._stack_row(
            form,
            "empty_holder",
            "Empty sample holder",
            "Empty sample holder stack, normalised against",
        )
        self._thickness_row(form)
        self._note(
            "One coefficient (cm⁻¹) per frame, from the frame means of both stacks."
        )

    def _build_t_cross_section(self) -> None:
        """Total microscopic cross section from a normalised transmission stack."""
        form = self._form()
        self._stack_row(form, "stack", "Stack", "Normalised transmission stack")

        self._row(
            form,
            "molar_mass",
            "Sample molar mass (g/mol)",
            dspin(1.0, 1e-6, 1e6, decimals=6, step=1.0),
        ).setToolTip("Molar mass of the sample, in grams per mole.")

        self._row(
            form,
            "density",
            "Effective density (g/cm³)",
            dspin(1.0, 1e-6, 1e3, decimals=6, step=0.1),
        ).setToolTip(
            "Effective density of the sample as measured, in g/cm³ : the "
            "packing of the sample, not the bulk density of the material."
        )
        self._thickness_row(form)
        self._note(
            "One cross section (barns) per frame. The stack must already be "
            "normalised against an open beam."
        )

    def _build_h_cross_section(self) -> None:
        """Hydrogen cross section of a compound mixture."""
        form = self._form()
        self._stack_row(form, "stack", "Stack", "Transmission stack of the sample")
        self._stack_row(
            form,
            "empty_holder",
            "Empty sample holder",
            "Empty sample holder stack, normalised against",
        )

        compounds = self._row(form, "compounds", "Compounds", QtWidgets.QLineEdit())
        compounds.setPlaceholderText("")
        compounds.setToolTip("Chemical formulae of the mixture, comma separated.")

        densities = self._row(
            form, "densities", "Densities (g/cm³)", QtWidgets.QLineEdit()
        )
        densities.setPlaceholderText("")
        densities.setToolTip("One density per compound, in the same order.")

        ratio = self._row(form, "ratio", "Ratio", QtWidgets.QLineEdit("1"))
        ratio.setToolTip(
            "Mixture ratio."
        )

        self._row(form, "by_volume", "Ratio by volume", QtWidgets.QCheckBox()).setToolTip(
            "Leave unticked for a ratio by mole."
        )

        self._thickness_row(form)
        self._note(
            "One cross section (barns) per frame. The sample stack must carry wavelengths."
        )

    ###########
    # RUNNERS #
    ###########

    def _on_run(self) -> None:
        """Run the selected function, reporting anything the form got wrong."""
        if self.job_running():
            return
        try:
            self._RUNNERS[self.current_function()](self)
        except ValueError as exc:
            popup(self, "Invalid Parameters", str(exc))

    def _selected_stack(self, role: str = "stack"):
        """Return the stack a combo has selected, or raise if it has none."""
        combo = self._widgets.get(role)
        stack = combo.currentData() if combo is not None else None
        if stack is None:
            raise ValueError("Select a stack.")
        return stack

    def _matched_stacks(self) -> tuple:
        """
        Return the (sample, empty sample holder) pair the coefficient
        computations take, checking they span the same number of frames.
        """
        stack = self._selected_stack("stack")
        holder = self._widgets["empty_holder"].currentData()
        if holder is None:
            raise ValueError("Select a stack and an empty sample holder.")
        if int(holder.data.shape[0]) != int(stack.data.shape[0]):
            raise ValueError(
                f"The empty sample holder has {int(holder.data.shape[0])} frame(s) "
                f"but the stack has {int(stack.data.shape[0])}.  Pick an empty "
                "sample holder with a matching frame count."
            )
        return stack, holder

    def _run_relative_attenuation(self) -> None:
        """Run relative attenuation on one stack's short/long bands."""
        stack = self._selected_stack()
        widgets = self._widgets
        n_slices = int(stack.data.shape[0])
        short_band = (int(widgets["sw0"].value()), int(widgets["sw1"].value()))
        long_band = (int(widgets["lw0"].value()), int(widgets["lw1"].value()))
        if not (
            0 <= short_band[0] < short_band[1] <= n_slices
            and 0 <= long_band[0] < long_band[1] <= n_slices
        ):
            raise ValueError(
                f"The short and long bands must lie within 0–{n_slices} frames "
                "with start < stop."
            )
        eps = float(widgets["eps"].value())

        if widgets["method"].currentText() == "Sum-of-Logs":
            # widgets snapshot for worker thread
            bin_factor = int(widgets["bin_factor"].value())
            filter_enabled = widgets["filter_enabled"].isChecked()
            filter_mode = widgets["filter_mode"].currentText()
            median_size = int(widgets["median_size"].value())
            gauss_sigma = float(widgets["gauss_sigma"].value())
            self._run_compute(
                "Rel. Attenuation (sum-of-logs)",
                lambda: sum_of_logs_relative_attenuation(
                    stack,
                    short_band,
                    long_band,
                    eps,
                    bin_factor,
                    filter_mode,
                    filter_enabled,
                    median_size,
                    gauss_sigma,
                ),
            )
        else:
            self._run_compute(
                "Relative Attenuation (images)",
                lambda: relative_attenuation(stack, short_band, long_band, eps),
            )

    def _run_atten_coefficient(self) -> None:
        """Run the coefficient computation against the empty sample holder."""
        stack, holder = self._matched_stacks()
        d_cm = float(self._widgets["d_cm"].value())
        self._run_compute(
            self.ATTENUATION_COEFFICIENT,
            lambda: atten_coefficient(stack, holder, d_cm),
        )

    def _run_t_cross_section(self) -> None:
        """Run the total microscopic cross section on one normalised stack."""
        stack = self._selected_stack()
        molar_mass = float(self._widgets["molar_mass"].value())
        density = float(self._widgets["density"].value())
        d_cm = float(self._widgets["d_cm"].value())
        self._run_compute(
            self.TOTAL_MICRO_CROSS_SECTION,
            lambda: t_cross_section(stack, molar_mass, density, d_cm),
        )

    def _run_h_cross_section(self) -> None:
        """Run the hydrogen cross section on a sample and its empty holder."""
        stack, holder = self._matched_stacks()
        compounds = parse_str_list(self._widgets["compounds"].text(), "Compounds")
        densities = parse_float_list(self._widgets["densities"].text(), "Densities")
        ratio = parse_float_list(self._widgets["ratio"].text(), "Ratio")
        if not (len(compounds) == len(densities) == len(ratio)):
            raise ValueError(
                f"Compounds ({len(compounds)}), densities ({len(densities)}) and "
                f"ratio ({len(ratio)}) must hold the same number of entries."
            )
        by_volume = self._widgets["by_volume"].isChecked()
        d_cm = float(self._widgets["d_cm"].value())
        self._run_compute(
            self.HYDROGEN_CROSS_SECTION,
            lambda: h_cross_section(
                stack, holder, compounds, densities, d_cm, ratio, by_volume
            ),
        )

    def _run_compute(self, name: str, fn) -> None:
        """
        Run an ROI process on a thread worker and emit the stack it produced.
        """
        if self.job_running():
            return
        self.run_job(
            lambda progress: fn(),
            name,
            lambda outputs: self._on_compute_result(name, outputs),
        )

    ##########
    # GATING #
    ##########

    def _set_job_busy(self, busy: bool) -> None:
        """Grey out the Run button when busy or with no stacks to work on."""
        enabled = bool(self._stacks) and not busy
        self._run_btn.setEnabled(enabled)
        self._run_btn.setToolTip("" if enabled else "No stacks loaded.")

    def _sync_run_enabled(self) -> None:
        """Update state of Run button according to job progress."""
        self._set_job_busy(self.job_running())

    ###################
    # RESULT HANDLERS #
    ###################

    def _on_compute_result(self, name: str, outputs: List) -> None:
        """Emit the stacks a compute job produced, and log their statistics."""
        for stack in outputs or []:
            self.stack_produced.emit(name, stack)

            if self._status is None:
                continue
            stats = stack.analysis_results()
            if stats:
                summary = ", ".join(f"{k}={stats[k]}" for k in list(stats)[:4])
                self._status.log(f"Compute '{name}': {summary}")
            self._status.log(f"Compute '{name}': added a new stack to the list.")

    ######################
    # FUNCTION DIRECTORY #
    ######################

    _BUILDERS: Dict[str, Callable] = {
        RELATIVE_ATTENUATION: _build_relative_attenuation,
        ATTENUATION_COEFFICIENT: _build_atten_coefficient,
        TOTAL_MICRO_CROSS_SECTION: _build_t_cross_section,
        HYDROGEN_CROSS_SECTION: _build_h_cross_section,
    }
    _RUNNERS: Dict[str, Callable] = {
        RELATIVE_ATTENUATION: _run_relative_attenuation,
        ATTENUATION_COEFFICIENT: _run_atten_coefficient,
        TOTAL_MICRO_CROSS_SECTION: _run_t_cross_section,
        HYDROGEN_CROSS_SECTION: _run_h_cross_section,
    }
    function_list = list(_BUILDERS)
