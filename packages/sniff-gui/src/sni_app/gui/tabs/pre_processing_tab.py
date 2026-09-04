"""
SNIFF: pre-processing panel.

Holds the pre-processing half of the Process tab.
"""

from pathlib import Path
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from sni_app.gui.panels.logging import StatusPanel
from sni_app.gui.panels.stack_store import StackStore
from sni_app.gui.panels.viewfinder import ImageWorkspace
from sni_app.gui.shared import (
    BTN_STYLE_RUN,
    PANEL_HEADER_STYLE,
    JobRunnerMixin,
    browse_btn,
    confirm,
    dspin,
    eliding_label,
    flat_btn,
    hbox,
    label,
    popup,
    run_btn,
    spin,
    vbox,
    vscroll,
)
from sni_app.gui.tabs.stack_stitcher import stitch_stacks
from superqt import QLabeledSlider, QRangeSlider

from sni_app.core.components.stack import Stack
from sni_app.core.io.stack import discover_and_load
from sni_app.core.process.roi_processes import roi_to_stack
from sni_app.core.process.stack_processes import (
    _OVERLAP_ROLES,
    stack_avg,
    stack_bin_frames,
    stack_join,
    stack_normalisation,
    stack_overlap_correction,
    stack_registration,
    stack_sbkg_correction,
    stack_scrubbing,
    stack_slice_acquisitions,
    stack_sum,
)
from sni_app.core.util.run_stats import (
    _first_shutter_count,
)


def is_single_frame(stack: Stack) -> bool:
    """Whether a stack holds exactly one frame (e.g. for black-body mask)."""
    return int(stack.data.shape[0]) == 1


class FunctionRunner(QtWidgets.QWidget):
    """
    Panel builder for processing functions interface. Contains a function runner for input parameters.

    Signals
    -------
    function_changed: the selected function changed (rebuild run-state)
    run_clicked: the Run button was pressed
    """

    function_changed = QtCore.pyqtSignal()
    run_clicked = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._stacks: List[tuple] = []  # Form: (name, Stack)
        self._selected: List[Stack] = []  # Stacks ticked for processing
        self._widgets: dict = {}  # Widgets with active parameters
        self._scale_edited = False  # Normalisation scale typed in by the user
        self._filling_scale = False  # guard, so autofill is not read as an edit

        v = vbox(self, (0, 0, 0, 0), 6)

        # Function dropdown
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(self.function_list)
        self._combo.setToolTip("Choose the processing function to run")
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        v.addWidget(self._combo)

        # Parameter panel
        self._panel = QtWidgets.QWidget()
        self._panel_layout = vbox(self._panel, (2, 2, 2, 2), 10)  # gap between params
        v.addWidget(self._panel)

        self._run_btn = run_btn(self.run_clicked.emit)
        v.addWidget(self._run_btn)

        self._rebuild_params()

    def get_current_function(self) -> str:
        """Return function currently selected in drop-down."""
        return self._combo.currentText()

    def set_function(self, name: str) -> None:
        """Select function in drop-down."""
        self._combo.setCurrentIndex(self._combo.findText(name))

    def set_stacks(self, stacks: List[tuple]) -> None:
        """Update the stack list and widgets with stack dropdowns."""
        self._stacks = list(stacks)
        self._rebuild_params()

    def set_selected_stacks(self, selected: List[Stack]) -> None:
        """
        Update which stacks are ticked for processing.

        Only parameters derived from the ticked stacks are refreshed (the normalisation
        scale and the join ordering), so the panel's other fields survive a
        change of selection.
        """
        self._selected = list(selected)
        function = self.get_current_function()
        if function == "Normalisation":
            self._autofill_normalisation_scale()
        elif function == "Join Stacks":
            self._fill_join_order()

    def set_run_enabled(self, enabled: bool, reason: str = "") -> None:
        """Enable/disable the Run button; show reason if disabled."""
        self._run_btn.setEnabled(enabled)
        self._run_btn.setToolTip("" if enabled else reason)

    # ── Data-dependent slider bounds ──────────────────────────────────────────

    def _min_frames(self) -> int:
        """Number of slices in the smallest loaded stack (0 if none)."""
        return min((int(s.data.shape[0]) for _, s in self._stacks), default=0)

    def _min_half_width(self) -> int:
        """Half the slice width of the smallest (narrowest) loaded stack."""
        return min((int(s.data.shape[-1]) // 2 for _, s in self._stacks), default=0)

    ###################
    # PARAMETER PANEL #
    ###################

    def _on_combo_changed(self, _name: str) -> None:
        """Rebuild the parameter panel for the new function and notify listeners."""
        self._rebuild_params()
        self.function_changed.emit()

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

    def _rebuild_params(self) -> None:
        """Rebuild the parameter panel by dispatching to the current function's builder."""
        self._clear_layout(self._panel_layout)
        self._widgets = {}
        self._scale_edited = False  # the field the user typed into is gone
        builder = self._BUILDERS.get(
            self.get_current_function(), FunctionRunner._build_none
        )
        builder(self)

    def _add_row(self, caption: str, widget: QtWidgets.QWidget) -> None:
        """Add a labelled parameter row (caption above *widget*) to the panel."""
        row = vbox(spacing=1)
        row.addWidget(label(caption, "font-size: 11px; color: #444;"))
        row.addWidget(widget)
        self._panel_layout.addLayout(row)

    def _add_note(self, text: str) -> None:
        """Add an italic, word-wrapped note line to the parameter panel."""
        self._panel_layout.addWidget(
            label(text, "color: #888; font-style: italic; font-size: 11px;", wrap=True)
        )

    def _file_row(self, role: str, caption: str) -> None:
        """
        Add a file-picker row to panel.
        """
        w = QtWidgets.QWidget()
        h = hbox(w, (0, 0, 0, 0), 4)
        field = eliding_label(
            style="border: 1px solid #bbb; padding: 2px; font-size: 10px;"
        )
        field.setMinimumHeight(22)
        h.addWidget(field, stretch=1)
        h.addWidget(browse_btn(lambda: self._pick_file(field, caption)))
        self._widgets[role] = field
        self._add_row(caption, w)

    def _pick_file(self, field, caption: str) -> None:
        """Open a file dialog and write the chosen path into field."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select {caption}", "", "Text files (*.txt *.csv);;All files (*)"
        )
        if path:
            field.setText(path)

    @staticmethod
    def _make_range_slider(maximum: int) -> QRangeSlider:
        """Return a horizontal [0, maximum] QRangeSlider spanning its full range."""
        rng = QRangeSlider(QtCore.Qt.Orientation.Horizontal)
        rng.setRange(0, maximum)
        rng.setValue((0, maximum))
        return rng

    @staticmethod
    def _make_labeled_slider(maximum: int) -> QLabeledSlider:
        """Return a horizontal [0, maximum] QLabeledSlider starting at 0."""
        sld = QLabeledSlider(QtCore.Qt.Orientation.Horizontal)
        sld.setRange(0, maximum)
        sld.setValue(0)
        return sld

    #####################
    # FUNCTION BUILDERS #
    #####################

    def _build_none(self) -> None:
        """Populate the panel for a function that takes no parameters."""
        self._add_note("No parameters.")

    def _build_overlap(self) -> None:
        """Overlap Correction: file overrides for the three run-table text files."""
        # Correction arrays discovered inside each stack's source folder (see
        # Stack.from_folder) are used by default; browsing to a file below
        # overrides the internal data for that role.
        n_internal = sum(
            1
            for _, s in self._stacks
            if getattr(s, "stack_meta", None) and s.stack_meta.get("run_meta")
        )
        self._add_note(
            f"Internal overlap data for {n_internal} stacks. Browse to override. "
            if n_internal
            else "No internal overlap data detected."
        )
        self._file_row("shutter_count", "_ShutterCount.txt")
        self._file_row("shutter_times", "_ShutterTimes.txt")
        self._file_row("spectra", "_Spectra.txt")

    def _build_normalisation(self) -> None:
        """Normalisation: open-beam stack plus window/scale controls."""
        combo = QtWidgets.QComboBox()
        for name, stack in self._stacks:
            combo.addItem(name, stack)
        self._widgets["open_beam"] = combo
        self._add_row("Open-beam stack", combo)

        self._widgets["window_half"] = self._make_labeled_slider(self._min_half_width())
        self._add_row("Normalisation window half-length", self._widgets["window_half"])
        self._widgets["sum_neighbourhood"] = spin(0, 0, 1_000_000)
        self._add_row(
            "Frame neighbourhood for summation", self._widgets["sum_neighbourhood"]
        )

        self._widgets["scale"] = dspin(1.0, 0.0, 1e9)
        self._widgets["scale"].setToolTip(
            "Filled in from the stacks' shutter counts where both are known; "
            "typing a value of your own stops it being overwritten."
        )
        self._add_row(
            "Scale (open-beam shutters/ experiment shutters)", self._widgets["scale"]
        )
        self._scale_note = label(
            "", "color: #888; font-style: italic; font-size: 11px;", wrap=True
        )
        self._panel_layout.addWidget(self._scale_note)

        # The scale follows the chosen open beam (and the ticked stacks) until
        # the user overrides it by hand.
        self._widgets["scale"].valueChanged.connect(self._on_scale_edited)
        combo.currentIndexChanged.connect(lambda _index: self._autofill_normalisation_scale())
        self._autofill_normalisation_scale()

    def _on_scale_edited(self, _value: float) -> None:
        """Stop autofilling the normalisation scale once the user has set it themselves."""
        if not self._filling_scale:
            self._scale_edited = True

    def _autofill_normalisation_scale(self) -> None:
        """
        Calculate and autofill normalisation field.
        """
        scale_widget = self._widgets.get("scale")
        if scale_widget is None or self._scale_edited:
            return

        open_beam = self._widgets["open_beam"].currentData()
        ob_count = (
            _first_shutter_count(open_beam.overlap_data())
            if open_beam is not None
            else None
        )
        experiment_count = None
        for stack in self._selected:
            experiment_count = _first_shutter_count(stack.overlap_data())
            if experiment_count:
                break

        if not ob_count or not experiment_count:
            self._scale_note.setText(
                "No shutter counts on the selected stacks; set the scale by hand."
            )
            return

        self._filling_scale = True
        try:
            scale_widget.setValue(ob_count / experiment_count)
        finally:
            self._filling_scale = False
        self._scale_note.setText(
            f"Scale from shutter counts: {ob_count:g} / {experiment_count:g}."
        )

    def _build_scrubbing(self) -> None:
        """
        Scrubbing Correction parameters.
        """
        combo = QtWidgets.QComboBox()
        combo.addItem("(interpolate from weights table)", None)
        for name, stack in self._stacks:
            if getattr(stack, "path", None) is not None:
                combo.addItem(name, str(stack.path))
        combo.setToolTip(
            "Correct every selected stack against this one open-beam folder, "
            "instead of the open beams the weights table pairs each acquisition with."
        )
        self._widgets["open_beam_dir"] = combo
        self._add_row("Open-beam folder", combo)
        if combo.count() == 1:
            self._add_note(
                "No stacks loaded from a folder, so none can be the open beam."
            )

    def _build_registration(self) -> None:
        """Stack Registration parameters: a reference-stack drop-down plus keypoint count."""
        combo = QtWidgets.QComboBox()
        for name, stack in self._stacks:
            combo.addItem(name, stack)
        combo.setToolTip(
            "Reference stack every selected frame is aligned onto. A multi-frame "
            "stack is collapsed to its mean frame."
        )
        self._widgets["reference"] = combo
        self._add_row("Reference stack (image)", combo)

        self._widgets["keypoints"] = spin(200, 50, 100_000)
        self._add_row("Number of keypoints to extract", self._widgets["keypoints"])

    def _build_sbkg(self) -> None:
        """Black-body correction parameters: a black-body mask drop-down."""
        combo = QtWidgets.QComboBox()
        for name, stack in self.single_frame_stacks():
            combo.addItem(name, stack)
        combo.setToolTip(
            "Single-image stack marking where the black bodies sat in the beam. "
            "Any non-zero pixel counts as masked."
        )
        self._widgets["bb_mask"] = combo
        self._add_row("Black-body mask (1-image stack)", combo)
        if combo.count() == 0:
            self._add_note("Load a single-image mask stack to enable this process.")

    def single_frame_stacks(self) -> List[tuple]:
        """
        Loaded stacks holding exactly one frame : the shape a mask must be.

        Read both to fill the mask drop-down and to gate Run, so the two cannot
        disagree about what counts as a mask.
        """
        return [(name, stack) for name, stack in self._stacks if is_single_frame(stack)]

    def _build_slicer(self) -> None:
        """Stack Slicer parameters: a range slider with a live readout."""
        n = self._min_frames()
        rng = self._make_range_slider(n)
        self._widgets["range"] = rng
        readout = label(f"start 0  :  stop {n}", "font-size: 10px; color: #444;")
        rng.valueChanged.connect(
            lambda v: readout.setText(f"start {v[0]}  →  stop {v[1]}")
        )
        self._add_row("Slice range (start : stop)", rng)
        self._panel_layout.addWidget(readout)

    def _build_bin_frames(self) -> None:
        """Bin Stack Frames parameters: bin factor, start image, optional High/Low energy ranges."""
        n = self._min_frames()
        self._widgets["bin_factor"] = self._make_labeled_slider(n)
        self._add_row("Binning factor", self._widgets["bin_factor"])
        self._widgets["start_img"] = self._make_labeled_slider(n)
        self._add_row("Index of starting image", self._widgets["start_img"])

        chk = QtWidgets.QCheckBox("Separate HE / LE energies (he_le)")
        self._widgets["he_le"] = chk
        self._panel_layout.addWidget(chk)

        # HE / LE range sliders, shown only when the checkbox is ticked
        he_box = QtWidgets.QWidget()
        he_layout = vbox(he_box, (0, 0, 0, 0), 5)
        for key, caption in (
            ("he_range", "High-energy range"),
            ("le_range", "Low-energy range"),
        ):
            self._widgets[key] = self._make_range_slider(n)
            he_layout.addWidget(label(caption, "font-size: 11px; color: #444;"))
            he_layout.addWidget(self._widgets[key])
        he_box.setVisible(False)
        chk.toggled.connect(he_box.setVisible)
        self._panel_layout.addWidget(he_box)

    def _build_join(self) -> None:
        """
        Join Stacks parmeters: an orderable list of stacks ticked for processing.
        """
        lst = QtWidgets.QListWidget()
        lst.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        lst.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        lst.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        lst.setToolTip("Order in which the selected stacks are joined")
        self._join_empty_lbl = None  # not built yet; the fill below skips it
        self._widgets["join_order"] = lst
        self._fill_join_order()
        self._add_row("Ordering (top = first):", lst)

        self._join_empty_lbl = label(
            "Tick at least two stacks to join.",
            "color: #888; font-style: italic; font-size: 11px;",
            wrap=True,
        )
        self._panel_layout.addWidget(self._join_empty_lbl)
        self._join_empty_lbl.setVisible(lst.count() < 2)

        btn_row = hbox()
        for text, delta in (("▲ Up", -1), ("▼ Down", 1)):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _=False, d=delta: self._move_join_item(d))
            btn_row.addWidget(btn)
        btn_row.addStretch()
        self._panel_layout.addLayout(btn_row)

    def _fill_join_order(self) -> None:
        lst = self._widgets.get("join_order")
        if lst is None:
            return
        # The rows already placed, in their current order, so a change of
        # selection does not throw away an ordering the user built by hand.
        placed = [
            lst.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(lst.count())
        ]
        ordered = [s for s in placed if any(s is sel for sel in self._selected)]
        ordered += [s for s in self._selected if not any(s is p for p in ordered)]

        lst.clear()
        for stack in ordered:
            item = QtWidgets.QListWidgetItem(stack.display_name())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, stack)
            lst.addItem(item)

        note = getattr(self, "_join_empty_lbl", None)
        if note is not None:
            note.setVisible(lst.count() < 2)

    def _move_join_item(self, delta: int) -> None:
        """Move the selected join-order row up (delta = -1) or down (+1)."""
        lst = self._widgets.get("join_order")
        if lst is None:
            return
        row = lst.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < lst.count()):
            return
        lst.insertItem(target, lst.takeItem(row))
        lst.setCurrentRow(target)

    def _ordered_join_selection(self, selected: List[Stack]) -> tuple:
        """
        Return (stacks, names): the selected stacks in the join-order
        list's order, with their display names.  Selected stacks missing from
        the widget (e.g. added after the panel was built) are appended at the end.
        """
        lst = self._widgets.get("join_order")
        ordered: List[Stack] = []
        names: List[str] = []
        if lst is not None:
            selected_ids = {id(s) for s in selected}
            for i in range(lst.count()):
                stack = lst.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                if id(stack) in selected_ids:
                    ordered.append(stack)
                    names.append(lst.item(i).text())
        for stack in selected:
            if not any(stack is s for s in ordered):
                ordered.append(stack)
                names.append("(unlisted stack)")
        return ordered, names

    def _build_stitching(self) -> None:
        """
        Stack Stitching parameters: pick the short and long stacks from two
        drop-downs. finishes by running the stitcher window.
        """
        if len(self._stacks) < 2:
            self._add_note("Load at least two stacks.")

        short_combo = QtWidgets.QComboBox()
        long_combo = QtWidgets.QComboBox()
        for name, stack in self._stacks:
            short_combo.addItem(name, stack)
            long_combo.addItem(name, stack)
        # Default the long drop-down to a different stack where possible.
        if long_combo.count() > 1:
            long_combo.setCurrentIndex(1)

        self._widgets["short_stack"] = short_combo
        self._widgets["long_stack"] = long_combo
        self._widgets["collimation_distance"] = dspin(
            0.0, 0.0, 1e9, 4
        )
        self._widgets["delay"] = dspin(0, 0.0, 1e9, 6)
        self._add_row("Short stack", short_combo)
        self._add_row("Long stack", long_combo)
        self._add_row("Acquisition delay", self._widgets["delay"])
        self._add_row("Collimation distance", self._widgets["collimation_distance"])

    def stitch_inputs(self) -> tuple:
        """Return the (short, long) Stacks chosen in the stitching drop-downs."""
        short_w = self._widgets.get("short_stack")
        long_w = self._widgets.get("long_stack")
        delay = self._widgets.get("delay").value()
        distance = self._widgets.get("collimation_distance").value()
        short = (
            short_w.currentData() if short_w is not None and short_w.count() else None
        )
        long = long_w.currentData() if long_w is not None and long_w.count() else None
        return short, long, delay, distance

    #############
    # EXECUTION #
    #############

    def prepare_job(self, selected: List[Stack]):
        """
        Build a background job running the current function on selected stacks.

        The job returns the new stacks with their processes applied, each core
        process function records its own lineage, so nothing is stamped here.
        Returns None for Stack Stitcher (which runs its own dialog).
        """
        spec_builder = self._JOBS.get(self.get_current_function())
        return spec_builder(self, selected) if spec_builder is not None else None

    #############
    # JOB SPECS #
    #############

    @staticmethod
    def _reporter(progress, label: str):
        """Adapt a job's progress(done, total, name) callback to a core callback."""
        return lambda done, total: progress(done, total, label)

    def _job_overlap(self, selected: List[Stack]):

        overrides = {role: self._widgets[role].text().strip() for role in _OVERLAP_ROLES}
        return lambda p: stack_overlap_correction(
            selected,
            progress_callback=self._reporter(p, "Overlap Correction"),
            **overrides,
        )

    def _job_normalisation(self, selected: List[Stack]):
        open_beam = self._widgets["open_beam"].currentData()
        if open_beam is None:
            raise ValueError("Choose an open-beam stack.")
        window_half = self._widgets["window_half"].value()
        sum_neighbourhood = self._widgets["sum_neighbourhood"].value()
        scale = self._widgets["scale"].value()
        return lambda p: stack_normalisation(
            selected,
            open_beam,
            window_half,
            sum_neighbourhood,
            scale,
            progress_callback=self._reporter(p, "Normalisation"),
        )

    def _job_registration(self, selected: List[Stack]):
        reference = self._widgets["reference"].currentData()
        if reference is None:
            raise ValueError("Choose a reference stack.")
        keypoints = self._widgets["keypoints"].value()
        return lambda p: stack_registration(selected, reference, keypoints=keypoints)

    def _job_scrubbing(self, selected: List[Stack]):
        combo = self._widgets.get("open_beam_dir")
        open_beam_dir = combo.currentData() if combo is not None else None
        weights = selected[0].stack_meta.get("weights_data_frame") if selected else None
        if weights is None and not open_beam_dir:
            raise ValueError(
                "No weights dataframe on the selected stack : load its "
                "folder via 'Load Stacks' first, or pick an open-beam folder."
            )
        return lambda p: stack_scrubbing(selected, weights, open_beam_dir=open_beam_dir)

    def _job_sbkg(self, selected: List[Stack]):
        bb_mask = self._widgets["bb_mask"].currentData()
        if bb_mask is None:
            raise ValueError("Choose a black-body mask stack.")
        return lambda p: stack_sbkg_correction(
            selected,
            bb_mask,
            progress_callback=self._reporter(p, "SBKG Correction"),
        )

    def _job_slicer(self, selected: List[Stack]):
        lo, hi = self._widgets["range"].value()
        return lambda p: stack_slice_acquisitions(selected, start=lo, stop=hi)

    def _job_avg(self, selected: List[Stack]):
        return lambda p: stack_avg(selected)

    def _job_sum(self, selected: List[Stack]):
        # stack_sum raises if the selected stacks don't share a shape.
        return lambda p: stack_sum(selected)

    def _job_bin_frames(self, selected: List[Stack]):
        bin_factor = self._widgets["bin_factor"].value()
        start_img = self._widgets["start_img"].value()
        he_le = False
        if self._widgets["he_le"].isChecked():
            he_le = (
                True,
                list(self._widgets["he_range"].value()),
                list(self._widgets["le_range"].value()),
            )
        return lambda p: stack_bin_frames(
            selected, bin_factor=bin_factor, start_img=start_img, he_le=he_le
        )

    def _job_join(self, selected: List[Stack]):
        ordered, _names = self._ordered_join_selection(selected)
        if len(ordered) < 2:
            raise ValueError("Select at least two stacks to join.")
        return lambda p: stack_join(ordered)

    ######################
    # FUNCTION DIRECTORY #
    ######################

    _BUILDERS = {
        "Overlap Correction": _build_overlap,
        "Normalisation": _build_normalisation,
        "Scrubbing Correction": _build_scrubbing,
        "SBKG Correction": _build_sbkg,
        "Stack Slicer": _build_slicer,
        "Stack Averaging": _build_none,
        "Stack Summation": _build_none,
        "Bin Stack Frames": _build_bin_frames,
        "Join Stacks": _build_join,
        "Stack Registration": _build_registration,
        "Stack Stitching": _build_stitching,
    }
    _JOBS = {
        "Overlap Correction": _job_overlap,
        "Normalisation": _job_normalisation,
        "Scrubbing Correction": _job_scrubbing,
        "SBKG Correction": _job_sbkg,
        "Stack Slicer": _job_slicer,
        "Stack Averaging": _job_avg,
        "Stack Summation": _job_sum,
        "Bin Stack Frames": _job_bin_frames,
        "Join Stacks": _job_join,
        "Stack Registration": _job_registration,
    }  # stack stitching needs its own logic
    function_list = list(_BUILDERS)


class PreProcessingPanel(JobRunnerMixin, QtWidgets.QWidget):
    """
    Pre-processing function runner.

    The column heading sits above the functions box, matching the Analysis
    tab's compute panel.

    Signals
    -------
    log_requested : a message to append to the status log (passes str)
    """

    log_requested = QtCore.pyqtSignal(str)

    # One-to-one processes
    _ABBREVIATIONS = {
        "Overlap Correction": "O.C.",
        "Normalisation": "Norm.",
        "Scrubbing Correction": "Scrub.",
        "SBKG Correction": "SBKG.",
        "Stack Slicer": "Slice.",
        "Stack Registration": "Reg.",
    }

    def __init__(
        self,
        store: StackStore,
        status: Optional["StatusPanel"] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._status = status

        self._runner = FunctionRunner()
        self._runner.function_changed.connect(self._update_run_state)
        self._runner.run_clicked.connect(self._on_run)

        outer = vbox(self, (0, 0, 0, 0), 0)
        outer.addWidget(label("Preprocessing Functions", PANEL_HEADER_STYLE))

        body = QtWidgets.QWidget()
        vb = vbox(body, (8, 8, 8, 8), 8)
        vb.addWidget(self._runner)
        vb.addStretch()  # keep the functions at the top of the column
        outer.addWidget(vscroll(body), stretch=1)

        store.stacks_changed.connect(self._refresh_runner_context)
        store.selection_changed.connect(self._refresh_runner_selection)
        store.busy_changed.connect(lambda _busy: self._update_run_state())
        self._refresh_runner_context()

    #############
    # SAVE/LOAD #
    #############

    def current_function(self) -> str:
        """Return the selected function's name, for saving in a project."""
        return self._runner.get_current_function()

    def set_function(self, name: str) -> None:
        """Restore the function drop-down from a project."""
        self._runner.set_function(name)

    #################
    # RUN EXECUTION #
    #################

    def _refresh_runner_context(self) -> None:
        """Provide runner with the loaded stacks."""
        # Selection first: rebuilding the parameters reads it back.
        self._runner.set_selected_stacks(self._store.selected_stacks())
        self._runner.set_stacks(self._store.pairs())
        self._update_run_state()

    def _refresh_runner_selection(self) -> None:
        """Provide runner with the stacks ticked for processing."""
        self._runner.set_selected_stacks(self._store.selected_stacks())
        self._update_run_state()

    def _set_job_busy(self, busy: bool) -> None:
        """Flag the shared store busy so every panel's controls gate together."""
        self._store.set_busy(busy)

    def _update_run_state(self) -> None:
        """
        Enable/disable the Run button: disabled while a background job runs,
        when no stacks are selected, or when a function's stack preconditions are not met.
        """
        fn = self._runner.get_current_function()
        n_selected = len(self._store.selected_stacks())

        enabled, reason = True, ""
        if self._store.is_busy():
            enabled, reason = False, "Another operation is still running."
        elif fn == "Stack Stitching":
            if self._store.count() < 2:
                enabled, reason = (
                    False,
                    "Load at least two stacks to stitch (short + long).",
                )
        elif fn == "Join Stacks" and n_selected < 2:
            enabled, reason = False, "Select at least two stacks to join."
        elif fn == "Stack Summation" and n_selected < 2:
            enabled, reason = False, "Select at least two stacks to sum (same shape)."
        elif fn == "SBKG Correction" and not self._runner.single_frame_stacks():
            enabled, reason = (
                False,
                "Load a single-image black-body mask stack first.",
            )
        elif n_selected == 0:
            enabled, reason = False, "Select at least one stack for processing."

        self._runner.set_run_enabled(enabled, reason)

    def _on_run(self) -> None:
        """Run the selected function on a worker; add its output stacks when done."""
        if self._store.is_busy():
            return
        fn = self._runner.get_current_function()

        if fn == "Stack Stitching":
            self._run_stitching()
            return

        selected = self._store.selected_stacks()
        if not selected:
            return

        try:
            job = self._runner.prepare_job(selected)
        except Exception as exc:
            self.log_requested.emit(f"Run '{fn}' failed: {exc}")
            popup(self, "Run Failed", f"{fn} failed:\n\n{exc}")
            return
        if job is None:
            return

        self.log_requested.emit(f"Run '{fn}' on {len(selected)} stack(s) ...")
        sources = [stack.display_name() for stack in selected]
        self.run_job(
            job,
            fn,
            lambda outputs, f=fn, s=sources: self._on_run_result(f, outputs, s),
        )

    def _on_run_result(self, fn: str, outputs, sources: List[str]) -> None:
        """
        Add the stacks produced by a processing worker to the store.

        A map-mode process returns one stack per input, in order, so each output
        is named after the stack it came from; the names chain as processes are
        applied in turn.
        """
        outputs = outputs or []
        abbreviation = self._ABBREVIATIONS.get(fn)
        for i, out in enumerate(outputs, 1):
            if abbreviation is not None and i <= len(sources):
                out.stack_meta["display_name"] = f"({abbreviation}) {sources[i - 1]}"
            else:
                out.stack_meta["display_name"] = f"{fn} #{i}"
        self._store.add_many(outputs)

        if outputs:
            self.log_requested.emit(
                f"Run '{fn}': produced {len(outputs)} new stack(s)."
            )
        else:
            self.log_requested.emit(f"Run '{fn}': produced no output stacks.")
            popup(self, fn, f"'{fn}' produced no output stacks.")

    def _run_stitching(self) -> None:
        """
        Launch interactive stacks stitcher and handle returned stack.
        """
        short, long, delay, distance = self._runner.stitch_inputs()
        if short is None or long is None:
            popup(
                self,
                "Stack Stitching",
                "Choose a short and a long stack from the drop-down menus.",
            )
            return
        if short is long:
            popup(
                self,
                "Stack Stitching",
                "Short and long stacks cannot be the same",
            )
            return

        self.log_requested.emit("Run 'Stack Stitching': launching stitcher...")
        try:
            stitched = stitch_stacks(short, long, delay, distance, parent=self)
        except Exception as exc:
            self.log_requested.emit(f"Run 'Stack Stitching' failed: {exc}")
            popup(self, "Run Failed", f"Stack Stitching failed:\n\n{exc}")
            return

        if stitched is None:
            self.log_requested.emit("Run 'Stack Stitching': cancelled.")
            return

        self._store.add(stitched, name="Stack Stitching #1")
        self.log_requested.emit("Run 'Stack Stitching': produced 1 new stack.")


class StackLoader(JobRunnerMixin, QtWidgets.QWidget):
    """
    Stack loading.

    Signals
    -------
    log_requested : string to append to status log.
    """

    log_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        store: StackStore,
        status: Optional["StatusPanel"] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._status = status  # Used for status bar
        self._source_dir = ""  # persisted in projects

    def source_directory(self) -> str:
        """Return the last source directory used, for saving in a project."""
        return self._source_dir

    def set_source_directory(self, path: str) -> None:
        """Restore the source directory from a project."""
        self._source_dir = path or ""

    ###########
    # LOADING #
    ###########

    def browse_and_load(self) -> None:
        """Prompt for a source directory and load the stack folders under it."""
        if self._store.is_busy():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Source Directory", self._source_dir
        )
        if not path:
            return
        self._source_dir = path
        self._load()

    def _load(self) -> None:
        """
        Acquire stacks for the store via the core reader.
        """
        if self._store.is_busy():
            return

        root = Path(self._source_dir)
        if not root.is_dir():
            popup(self, "Invalid Folder", f"Not a directory:\n  {root}")
            return

        self.log_requested.emit(f"Load Stacks: reading '{root}' ...")

        def job(progress) -> list:
            """Read all stack folders under root, reporting frame-by-frame progress."""
            stacks, _ = discover_and_load(
                root,
                progress_callback=lambda done, total: progress(
                    done, total, "Loading stacks"
                ),
            )
            return stacks

        self.run_job(job, "Loading stacks", self._on_load_result)

    def _on_load_result(self, stacks: list) -> None:
        """Add the stacks read by the load worker to the store (GUI thread)."""
        added = self._store.add_many(stacks)

        if added == 0:
            self.log_requested.emit("Load Stacks: no valid stack folders found.")
        else:
            weight_msg = (
                "weights available"
                if self._store.weights_available()
                else "no open-beam weights found"
            )
            self.log_requested.emit(
                f"Load Stacks: added {added} stack(s) "
                f"(total {self._store.count()}) : {weight_msg}."
            )

    ##########
    # GATING #
    ##########

    def _set_job_busy(self, busy: bool) -> None:
        """Flag the shared store busy so every panel's controls gate together."""
        self._store.set_busy(busy)


RoiXYWH = tuple[int, int, int, int]


class RoiToolsPanel(QtWidgets.QWidget):
    """
    ROI to stack operator plus ROI memory system.

    Signals
    -------
    log_requested : string to append to the status log
    """

    log_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        store: StackStore,
        workspace: ImageWorkspace,
        status: Optional[object] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._workspace = workspace
        self._status = status
        self._saved: dict[str, RoiXYWH] = {}
        self._rows: List[QtWidgets.QWidget] = []

        outer = vbox(self, (0, 0, 0, 0), 6)
        outer.addWidget(self._build_crop_box())
        outer.addWidget(self._build_memory_box())

    ###################
    # WIDGET BUILDERS #
    ###################

    def _build_crop_box(self) -> QtWidgets.QGroupBox:
        """Build the "ROI to Stack" group with its cropping button."""
        box = QtWidgets.QGroupBox("ROI to Stack")
        box.setStyleSheet("QGroupBox { font-weight: bold; }")
        lay = vbox(box, (8, 6, 8, 8), 5)
        lay.addWidget(
            label(
                "Crop selected stacks to ROI and add copies to Stack list.",
                "color: #555; font-weight: normal; font-size: 11px;",
                wrap=True,
            )
        )
        self._crop_btn = QtWidgets.QPushButton("Create ROI Stack(s)")
        self._crop_btn.setStyleSheet(BTN_STYLE_RUN)
        self._crop_btn.setToolTip("Crop to ROI.")
        self._crop_btn.clicked.connect(self._roi_to_stack)
        lay.addWidget(self._crop_btn)
        return box

    def _build_memory_box(self) -> QtWidgets.QGroupBox:
        """Build the "Saved ROIs" memory group (nickname entry + saved list)."""
        box = QtWidgets.QGroupBox("Saved ROIs")
        box.setStyleSheet("QGroupBox { font-weight: bold; }")
        lay = vbox(box, (8, 6, 8, 8), 5)

        row = hbox(spacing=4)
        self._name_field = QtWidgets.QLineEdit()
        self._name_field.setPlaceholderText("ROI name...")
        self._name_field.returnPressed.connect(self._save_current_roi)
        row.addWidget(self._name_field, stretch=1)
        save_btn = QtWidgets.QPushButton("Save ROI")
        save_btn.setToolTip("Save ROI under given name.")
        save_btn.clicked.connect(self._save_current_roi)
        row.addWidget(save_btn)
        lay.addLayout(row)

        body = QtWidgets.QWidget()
        self._list_layout = vbox(body, (2, 2, 2, 2), 2)
        self._empty_lbl = label(
            "No saved ROIs.", "color: #888; font-style: italic; font-size: 11px;"
        )
        self._list_layout.addWidget(self._empty_lbl)
        self._list_layout.addStretch()
        scroll = vscroll(body)
        scroll.setMinimumHeight(90)
        scroll.setMaximumHeight(220)
        lay.addWidget(scroll)
        return box

    ##################
    # ROI OPERATIONS #
    ##################

    def _roi_to_stack(self) -> None:
        """Crop every ticked stack to the current ROI and add the crops to the store."""
        selected = self._store.selected_stacks()
        if not selected:
            popup(
                self,
                "No Stacks Ticked",
                "Tick one or more stacks in the Stacks list to crop to the ROI.",
            )
            return

        x, y, w, h = self._workspace.current_roi_xywh()
        crops = roi_to_stack(selected, (x, y, w, h))
        for source, crop in zip(selected, crops):
            # The ROI is clamped per stack, so name each crop by its own size.
            crop_h, crop_w = crop.data.shape[1:]
            self._store.add(
                crop, name=f"{source.display_name()} [ROI {crop_w}×{crop_h}]"
            )

        self.log_requested.emit(
            f"ROI to Stack: added {len(crops)} cropped stack(s) from ROI "
            f"(x={x}, y={y}, w={w}, h={h})."
        )

    def _save_current_roi(self) -> None:
        """Store the current ROI box under the nickname in the entry field."""
        name = self._name_field.text().strip()
        if not name:
            popup(self, "No Nickname", "Enter a nickname for the ROI first.")
            return
        if name in self._saved and not confirm(
            self, "Overwrite ROI", f"Replace the saved ROI '{name}'?"
        ):
            return
        roi = self._workspace.current_roi_xywh()
        self._saved[name] = roi
        self._name_field.clear()
        self._rebuild_rows()
        x, y, w, h = roi
        self.log_requested.emit(f"Saved ROI '{name}' (x={x}, y={y}, w={w}, h={h}).")

    def _load_roi(self, name: str) -> None:
        """Re-apply a saved ROI to the previewer."""
        roi = self._saved.get(name)
        if roi is None:
            return
        self._workspace.set_roi_xywh(*roi)
        x, y, w, h = roi
        self.log_requested.emit(f"Applied ROI '{name}' (x={x}, y={y}, w={w}, h={h}).")

    def _remove_roi(self, name: str) -> None:
        """Forget a saved ROI."""
        if name in self._saved:
            del self._saved[name]
            self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Recreate the saved-ROI list rows from the stored ROIs."""
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []

        for name, (x, y, w, h) in self._saved.items():
            row = QtWidgets.QWidget()
            rl = hbox(row, (0, 0, 0, 0), 4)
            text = eliding_label(
                f"{name}  ({x},{y},{w}×{h})",
                QtCore.Qt.TextElideMode.ElideRight,
                "font-size: 11px;",
            )
            text.setToolTip(f"{name}: x={x}, y={y}, w={w}, h={h}")
            rl.addWidget(text, stretch=1)

            load_btn = QtWidgets.QPushButton("Load")
            load_btn.setFixedHeight(22)
            load_btn.setToolTip("Apply this ROI to the previewer")
            load_btn.clicked.connect(lambda _=False, n=name: self._load_roi(n))
            rl.addWidget(load_btn)

            del_btn = flat_btn(
                "✕",
                "Delete this saved ROI",
                "QPushButton { border: none; color: #aa3333; font-weight: bold; }"
                "QPushButton:hover { color: #ff4444; }",
                width=22,
            )
            del_btn.clicked.connect(lambda _=False, n=name: self._remove_roi(n))
            rl.addWidget(del_btn)

            # Insert before the trailing stretch so rows stay packed at the top.
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._rows.append(row)

        self._empty_lbl.setVisible(not self._saved)

    ###################
    #   PERSISTENCE   #
    ###################

    def saved_rois(self) -> dict[str, List[int]]:
        """Return the saved ROIs as JSON-friendly ``{name: [x, y, w, h]}`` for projects."""
        return {name: [int(v) for v in roi] for name, roi in self._saved.items()}

    def set_saved_rois(self, data: Optional[dict]) -> None:
        """Restore saved ROIs from a project's ``{name: [x, y, w, h]}`` mapping."""
        self._saved = {}
        if isinstance(data, dict):
            for name, roi in data.items():
                try:
                    x, y, w, h = (int(v) for v in roi)
                except (TypeError, ValueError):
                    continue
                self._saved[str(name)] = (x, y, w, h)
        self._rebuild_rows()
