import gc
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from sni_app.core.components.stack import Stack
from sni_app.core.io.stack import export_stacks
from sni_app.core.util.scrubbing import _merge_weights
from sni_app.gui.shared import (
    BTN_STYLE_RED,
    JobRunnerMixin,
    array_to_pixmap,
    confirm,
    confirm_optout,
    eliding_label,
    flat_btn,
    hbox,
    label,
    popup,
    vbox,
    vscroll,
)

# Save-dialog filters for a single stack, and the extension each one writes.
EXPORT_FILTERS = {
    "FITS files, one per frame (*.fits)": ".fits",
    "3D TIFF, one file (*.tiff)": ".tiff",
}


class StackStore(QtCore.QObject):
    """
    Set of loaded stacks, shared among all relevant processes.
    The loaded stacks, shared by every panel that shows or works on them.
    Records stack naming and selection.
    """

    stacks_changed = QtCore.pyqtSignal()
    selection_changed = QtCore.pyqtSignal()
    busy_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._stacks: List[Stack] = []
        self._weights_df = None
        self._busy = False
        self._confirm_removals = True

    ##########
    # STATES #
    ##########

    @staticmethod
    def is_selected(stack: Stack) -> bool:
        """Return whether stack is selected for processing."""
        meta = getattr(stack, "stack_meta", None) or {}
        return bool(meta.get("selected_for_processing", False))

    #############
    #  GETTERS  #
    #############

    def stacks(self) -> List[Stack]:
        """Return every loaded stack, in list order."""
        return list(self._stacks)

    def pairs(self) -> List[tuple]:
        """Return list of loaded (name, Stack)."""
        return [(s.display_name(), s) for s in self._stacks]

    def selected_stacks(self) -> List[Stack]:
        """Return the stacks ticked for processing."""
        return [s for s in self._stacks if self.is_selected(s)]

    def selected_pairs(self) -> List[tuple]:
        """Return list of processing-selected (name, Stack)."""
        return [(s.display_name(), s) for s in self._stacks if self.is_selected(s)]

    def count(self) -> int:
        """Return the number of loaded stacks."""
        return len(self._stacks)

    def weights_available(self) -> bool:
        """Return whether a non-empty scrubbing-weights dataframe is loaded."""
        df = self._weights_df
        return df is not None and not bool(getattr(df, "empty", True))

    def is_busy(self) -> bool:
        """Return whether a panel's background job is running."""
        return self._busy

    def confirm_removals(self) -> bool:
        """
        Return whether removing a single stack should still be confirmed.
        """
        return self._confirm_removals

    #############
    #  SETTERS  #
    #############

    def add(
        self, stack: Stack, name: Optional[str] = None, selected: Optional[bool] = None
    ) -> None:
        """Append stack."""
        self._add(stack, name, selected)
        self.stacks_changed.emit()

    def add_many(self, stacks: Iterable[Stack]) -> int:
        """Append several stacks, emitting the change once. Returns how many were added."""
        added = 0
        for stack in stacks:
            self._add(stack)
            added += 1
        if added:
            self.stacks_changed.emit()
        return added

    def replace_all(self, stacks: Iterable[Stack]) -> None:
        """Purge and repopulate store (used for project loading)."""
        self._stacks = []
        self._weights_df = None
        for stack in stacks:
            self._add(stack)
        self.stacks_changed.emit()
        gc.collect()

    def remove(self, stack: Stack) -> None:
        """Drop stack from the list to free it from memory."""
        self._stacks = [s for s in self._stacks if s is not stack]
        self.stacks_changed.emit()
        gc.collect()

    def clear(self) -> None:
        """Drop all stacks and weights."""
        self._stacks = []
        self._weights_df = None
        self.stacks_changed.emit()
        gc.collect()

    def rename(self, stack: Stack, name: str) -> None:
        """change stack display name."""
        stack.stack_meta["display_name"] = name
        self.stacks_changed.emit()

    def set_selected(self, stack: Stack, selected: bool) -> None:
        """Tick or untick stack for processing."""
        stack.stack_meta["selected_for_processing"] = bool(selected)
        self.selection_changed.emit()

    def set_confirm_removals(self, confirm_removals: bool) -> None:
        """Ask (or stop asking) before a stack is removed from the list."""
        self._confirm_removals = bool(confirm_removals)

    def set_busy(self, busy: bool) -> None:
        """Flag a background job to panel controls as running or done."""
        busy = bool(busy)
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def _add(
        self, stack: Stack, name: Optional[str] = None, selected: Optional[bool] = None
    ) -> None:
        """Record stack and state without emitting the change."""
        meta = stack.stack_meta

        stack.robust_stack_uuid()
        meta["display_name"] = name if name is not None else stack.display_name()
        if selected is not None:
            meta["selected_for_processing"] = bool(selected)
        else:
            meta.setdefault("selected_for_processing", False)
        self._weights_df = _merge_weights(
            self._weights_df, meta.get("weights_data_frame")
        )
        self._stacks.append(stack)


class StackItem(QtWidgets.QWidget):
    """
    Row inside the stacks panel: preview of stack with display name, metadata, and options for seleaction and naming.
    Changes are emitted to the panel, which interfaces with the stack store.
    In the context menu, stacks can be renamed, deleted, and exported.
    """

    remove_requested = QtCore.pyqtSignal(object)
    export_requested = QtCore.pyqtSignal(object)
    selected = QtCore.pyqtSignal(object)
    processing_toggled = QtCore.pyqtSignal(object, bool)
    rename_requested = QtCore.pyqtSignal(object, str)

    def __init__(self, data: Stack, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._stack = data
        name = data.display_name()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
        )  # Highlight and change cursor on mouse over
        self.setStyleSheet(
            "StackItem { border-radius: 4px; }StackItem:hover { background: #eaf2fb; }"
        )

        layout = vbox(self, (4, 3, 4, 3), 1)
        header = hbox(spacing=4)  # preview, stack name, buttons

        # Stack preview (uses frame average)
        self._icon_lbl = QtWidgets.QLabel()
        self._icon_lbl.setFixedSize(30, 30)
        self._icon_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        frame = np.mean(data.data, axis=0) if data.data.ndim >= 3 else data.data
        pix = array_to_pixmap(frame, 28)
        if pix is not None:
            self._icon_lbl.setPixmap(pix)
        header.addWidget(self._icon_lbl)

        # Stack name (elides)
        self._name_lbl = eliding_label(
            name, QtCore.Qt.TextElideMode.ElideRight, "font-weight: bold;"
        )
        self._name_lbl.setToolTip(str(data.path) if data.path else "In-memory Stack")
        header.addWidget(self._name_lbl, stretch=1)

        rename_btn = flat_btn(
            "✎",
            "Rename stack",
            "QPushButton { border: none; }QPushButton:hover { color: #0055aa; }",
        )
        rename_btn.clicked.connect(self._rename)
        header.addWidget(rename_btn)

        # Checkbox for stack processing
        self._select = QtWidgets.QCheckBox()
        self._select.setChecked(StackStore.is_selected(data))
        self._select.setToolTip("Select this stack for processing")
        self._select.toggled.connect(
            lambda checked: self.processing_toggled.emit(self._stack, checked)
        )
        header.addWidget(self._select)

        remove_btn = flat_btn(
            "x",
            "Remove this stack from the list",
            "QPushButton { border: none; color: #aa3333; font-weight: bold; }"
            "QPushButton:hover { color: #ff4444; }",
        )
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._stack))
        header.addWidget(remove_btn)

        layout.addLayout(header)

        # Stack info (subheader)
        path_str = "(in-memory)" if data.path is None else str(data.path)
        self._info_lbl = eliding_label(
            f"Path: {path_str}; Shape: {data.data.shape}",
            style="color: #777; font-size: 10px;",
        )
        layout.addWidget(self._info_lbl)

        self.setToolTip(
            f"{name}\n\nProcesses:\n{data.process_history_string()}"
        )  # Pack history into tooltip

    def get_stack(self) -> Stack:
        """Return the Stack this row represents."""
        return self._stack

    def set_checked(self, checked: bool) -> None:
        """Re-tick the checkbox from the store (does not announce)."""
        if self._select.isChecked() == checked:
            return
        self._select.blockSignals(True)
        self._select.setChecked(checked)
        self._select.blockSignals(False)

    def _rename(self) -> None:
        """Prompt new display name and request it from the panel."""
        new, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename Stack",
            "New display name:",
            text=self._stack.display_name(),
        )
        if ok and new.strip():
            self.rename_requested.emit(self._stack, new.strip())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Emits request for preview of stack."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.selected.emit(self._stack)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Offer the row's rename/export/remove actions on right-click."""
        if self._stack is None:  # row released while the menu was opening
            return
        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction("Rename Stack...")
        export_action = menu.addAction("Export Stack...")
        menu.addSeparator()
        remove_action = menu.addAction("Remove Stack")
        chosen = menu.exec(event.globalPos())
        if chosen is rename_action:
            self._rename()
        elif chosen is export_action:
            self.export_requested.emit(self._stack)
        elif chosen is remove_action:
            self.remove_requested.emit(self._stack)

    def release(self) -> None:
        """Drop the stack reference from memory"""
        self._stack = None
        self._icon_lbl.clear()


class StackListPanel(JobRunnerMixin, QtWidgets.QGroupBox):
    """
    List of stacks with "Export Selected" and "Clear Stacks" buttons.

    Exporting the whole ticked selection is the window's job (it owns the export
    box), so the button only asks for it; exporting one stack, offered on a
    row's right-click menu, is written straight from here.
    """

    stack_selected = QtCore.pyqtSignal(str, object)
    export_selected_requested = QtCore.pyqtSignal()
    log_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        store: StackStore,
        status=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__("Stacks", parent)

        self._store = store
        self._status = status
        self._rows: List[StackItem] = []

        self.setStyleSheet("QGroupBox { font-weight: bold; }")
        sb_layout = vbox(self, (4, 4, 4, 4))

        # Scrollable stacks list
        stacks_body = QtWidgets.QWidget()
        self._stacks_layout = vbox(stacks_body, (2, 2, 2, 2), 2)
        self._stacks_empty_lbl = label(
            "No Stacks.", "color: #888; font-style: italic; font-size: 11px;"
        )
        self._stacks_layout.addWidget(self._stacks_empty_lbl)
        self._stacks_layout.addStretch()
        sb_layout.addWidget(vscroll(stacks_body))

        # Export / Clear footer
        footer = hbox(margins=(0, 0, 0, 0), spacing=6)
        footer.addStretch()
        self._export_btn = QtWidgets.QPushButton("Export Selected")
        self._export_btn.clicked.connect(self.export_selected_requested.emit)
        footer.addWidget(self._export_btn)
        self._clear_btn = QtWidgets.QPushButton("Clear Stacks")
        self._clear_btn.setStyleSheet(BTN_STYLE_RED)
        self._clear_btn.setToolTip("Clear stacks list (cannot be undone)")
        self._clear_btn.clicked.connect(self._clear_stacks_confirm)
        footer.addWidget(self._clear_btn)
        sb_layout.addLayout(footer)

        store.stacks_changed.connect(self._rebuild_rows)
        store.selection_changed.connect(self._sync_row_checkboxes)
        store.selection_changed.connect(self._sync_export_enabled)
        self._rebuild_rows()

    #################################
    #         STORE GETTERS         #
    #################################

    def _rebuild_rows(self) -> None:
        """Recreate every row from the store (its contents or names changed)."""
        for row in self._rows:
            row.release()
            row.setParent(None)
            row.deleteLater()
        self._rows = []

        for stack in self._store.stacks():
            row = StackItem(stack)
            row.remove_requested.connect(self._remove_stack)
            row.export_requested.connect(self._export_stack)
            row.selected.connect(self._on_row_selected)
            row.processing_toggled.connect(self._store.set_selected)
            row.rename_requested.connect(self._store.rename)
            self._rows.append(row)
            # Insert before the trailing stretch so rows stay packed at the top
            self._stacks_layout.insertWidget(self._stacks_layout.count() - 1, row)

        self._stacks_empty_lbl.setVisible(not self._rows)
        self._sync_export_enabled()

    def _sync_row_checkboxes(self) -> None:
        """Re-tick every row from the store after a selection change anywhere."""
        for row in self._rows:
            row.set_checked(self._store.is_selected(row.get_stack()))

    #################################
    #         STORE SETTERS         #
    #################################

    def _on_row_selected(self, stack: Stack) -> None:
        """Notify previewer of stack selection."""
        self.stack_selected.emit(stack.display_name(), stack)

    def _remove_stack(self, stack: Stack) -> None:
        """
        Remove stack from the store, after confirmation unless the user has
        ticked "don't ask again" on an earlier removal.
        """
        if self._store.confirm_removals():
            remove, opted_out = confirm_optout(
                self, "Remove Stack", f"Remove stack '{stack.display_name()}'?"
            )
            if not remove:
                return
            if opted_out:
                self._store.set_confirm_removals(False)
                self.log_requested.emit(
                    "Stack removals will no longer ask for confirmation."
                )
        self._store.remove(stack)

    def _clear_stacks_confirm(self) -> None:
        """Clear the store after confirmation."""
        if self._rows and confirm(
            self, "Clear Stacks", "Remove all loaded stack(s)?\nThis cannot be undone."
        ):
            self._store.clear()
            self.log_requested.emit("Cleared all stacks from memory.")

    #################################
    #            EXPORT             #
    #################################

    def _sync_export_enabled(self) -> None:
        """Gate "Export Selected" on there being a selection to export."""
        n = len(self._store.selected_pairs())
        enabled = n > 0 and not self.job_running()
        self._export_btn.setEnabled(enabled)
        self._export_btn.setToolTip(
            f"Export the {n} stack(s) ticked for processing"
            if n
            else "Tick the stacks to export first."
        )

    def _set_job_busy(self, busy: bool) -> None:
        """Gate this panel's own buttons while a single-stack export runs."""
        self._clear_btn.setEnabled(not busy)
        self._sync_export_enabled()

    def _export_stack(self, stack: Stack) -> None:
        """Prompt for a destination and write one stack to it, off the GUI thread."""
        if self.job_running():
            popup(self, "Export Busy", "An export is already running.")
            return
        name = stack.display_name()
        path, chosen_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Export Stack '{name}'",
            name,
            ";;".join(EXPORT_FILTERS),
        )
        if not path:
            return

        destination = Path(path)
        # The dialog need not append one, and a typed suffix beats the filter.
        ext = destination.suffix.lower() or EXPORT_FILTERS.get(chosen_filter, ".fits")
        op_name = f"Exporting '{name}'"

        def job(progress) -> tuple:
            return export_stacks(
                [(name, stack)],
                destination.stem,
                ext,
                destination.parent,
                True,  # the save dialog has already settled the destination
                progress_callback=lambda i, n: progress(i, n, op_name),
            )

        target = destination.with_name(destination.stem + ext)
        self.run_job(
            job, op_name, lambda result: self._on_stack_exported(result, name, target)
        )

    def _on_stack_exported(self, result: tuple, name: str, target: Path) -> None:
        """Report the outcome of a single-stack export."""
        written, errors = result
        for line in errors:
            self.log_requested.emit(f"Export of {line}")
        if written:
            self.log_requested.emit(f"Exported stack '{name}' to '{target}'.")
        else:
            popup(
                self,
                "Export Failed",
                f"Could not export '{name}':\n\n" + "\n".join(errors),
            )
