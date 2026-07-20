"""
Full-Processing tab.
Builds a Directed Acyclic graph to display stack workflow history. Allows workflow replay with any compatible stacks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from sni_app.core.components.workflow import (
    WorkflowGraph,
    WorkflowNode,
    default_entry_map,
    replay_workflow,
    PROCESS_REGISTRY,
)
from sni_app.core.io.stack import safe_file_stem
from sni_app.core.io.workflow import load_workflow, save_workflow
from sni_app.gui.panels.stack_store import StackStore
from sni_app.gui.shared import hbox, label, popup, vbox
from superqt import QLabeledSlider

# Canvas zoom, as a percentage of the graph's natural size.
ZOOM_MIN_PCT = 25
ZOOM_MAX_PCT = 400
ZOOM_DEFAULT_PCT = 100


class ReplayDialog(QtWidgets.QDialog):
    """Ask which stack to feed in at each entry point of a workflow."""

    def __init__(
        self,
        graph: WorkflowGraph,
        store: StackStore,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Replay Workflow")
        self._combos: Dict[str, QtWidgets.QComboBox] = {}

        layout = vbox(self, (12, 12, 12, 12), 8)
        layout.addWidget(
            label(
                "Choose a stack to feed in at each entry point.",
                "color: #555;",
                wrap=True,
            )
        )

        form = QtWidgets.QFormLayout()
        form.setSpacing(6)
        pairs = store.pairs()
        for node in graph.entry_points():
            combo = QtWidgets.QComboBox()
            for name, stack in pairs:
                combo.addItem(name, stack)
            self._select_default(combo, node.stack)
            self._combos[node.uuid] = combo
            form.addRow(node.name + ":", combo)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _select_default(combo: QtWidgets.QComboBox, stack) -> None:
        """Preselect the combo entry backed by *stack*, if it is present."""
        if stack is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i) is stack:
                combo.setCurrentIndex(i)
                return

    def entry_map(self) -> Dict[str, object]:
        """Map each entry-point id to the chosen stack."""
        return {
            eid: c.currentData()
            for eid, c in self._combos.items()
            if c.currentData() is not None
        }


class FullProcessingTab(QtWidgets.QWidget):
    """Workflow tree + replay controls, bound to the shared stack store."""

    log_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        store: StackStore,
        status=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._status = status

        layout = vbox(self, (0, 0, 0, 0), 0)

        self._view = WorkflowView()
        self._view.replay_requested.connect(self._on_replay_requested)
        self._view.export_requested.connect(self._on_export_requested)
        self._view.import_requested.connect(self.import_workflow)
        layout.addWidget(self._view, stretch=1)

        store.stacks_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild the workflow view from the store's current stacks."""
        self._view.set_stacks(self._store.stacks())

    def _log(self, message: str) -> None:
        self.log_requested.emit(message)
        if self._status is not None:
            self._status.log(message)

    #####################
    # WORKFLOW FILES    #
    #####################

    def export_workflow(self) -> None:
        """Write the workflow on screen to a file (also driven from the File menu)."""
        graph = self._view.current_graph()
        if graph is None or not graph.nodes:
            popup(
                self,
                "Nothing to Export",
                "There is no workflow to export yet. Process some stacks first.",
            )
            return
        self._on_export_requested(graph)

    def import_workflow(self) -> None:
        """Read a workflow file and add it to the picker (also driven from the File menu)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Workflow",
            "",
            f"JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            graph = load_workflow(path)
        except (OSError, ValueError) as exc:
            popup(self, "Import Failed", f"Could not read that workflow:\n\n{exc}")
            return

        self._view.add_imported(graph)
        self._log(
            f"Imported workflow '{graph.label()}' from {path}. "
            f"Press Replay Workflow to run it on your stacks."
        )

    def _on_export_requested(self, graph: WorkflowGraph) -> None:
        """Prompt for a destination and write workflow."""
        suggested = safe_file_stem(graph.name or "workflow") + ".json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Workflow",
            suggested,
            f"SNIFF Workflow (JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            written = save_workflow(path, graph, name=graph.name or Path(path).stem)
        except (OSError, ValueError, TypeError) as exc:
            popup(self, "Export Failed", f"Could not write that workflow:\n\n{exc}")
            return
        self._log(f"Exported workflow ({graph.size()} steps) to {written}")

    def _on_replay_requested(self, graph: WorkflowGraph) -> None:
        """Prompt for entry stacks, replay the workflow, and store the results."""
        if graph is None or not graph.nodes:
            return
        if not self._store.stacks():
            # An imported recipe carries no data, so there is nothing to feed it.
            popup(
                self,
                "No Stacks Loaded",
                "Load the stacks you want to run this workflow on, then replay it.",
            )
            return
        dialog = ReplayDialog(graph, self._store, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        entry_map = dialog.entry_map() or default_entry_map(graph)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            result = replay_workflow(graph, entry_map)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            popup(self, "Replay Failed", f"Could not replay workflow:\n\n{exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        produced = result.new_stacks()
        for stack in produced:
            name = stack.stack_meta.get("display_name", "stack")
            stack.stack_meta["display_name"] = f"{name} (replay)"
        added = self._store.add_many(produced)

        self._log(
            f"Replayed workflow: reproduced {added} stack(s) from "
            f"{len(entry_map)} entry point(s)."
        )
        if result.errors:
            skipped = "\n".join(
                f"• {graph.nodes[nid].name}: {msg}"
                for nid, msg in result.errors.items()
                if nid in graph.nodes
            )
            popup(
                self,
                "Replay : some steps skipped",
                f"{added} stack(s) reproduced. These could not be replayed:\n\n{skipped}",
            )


NODE_W = 190
NODE_H = 66
COL_GAP = 70
ROW_GAP = 26

_ENTRY_BG = "#cfe8cf"
_ENTRY_BORDER = "#4f9e4f"
_PROC_BG = "#cfe2f3"
_PROC_BORDER = "#4a7fae"
_GHOST_BG = "#ededed"
_GHOST_BORDER = "#b3b3b3"
_UNREPLAYABLE_BORDER = "#c98a2b"


def _fmt_params(params: dict, limit: int = 3) -> str:
    """One-line summary of a params dict, truncated to a given number of entries."""
    if not params:
        return ""
    parts = []
    for k, v in list(params.items())[:limit]:
        if isinstance(v, float):
            v = f"{v:g}"
        parts.append(f"{k}={v}")
    if len(params) > limit:
        parts.append("...")
    return ", ".join(parts)


class NodeItem(QtWidgets.QGraphicsItem):
    """A single stack drawn as a rounded, labelled box."""

    def __init__(self, node: WorkflowNode) -> None:
        super().__init__()
        self.node = node
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setToolTip(self._tooltip())

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, NODE_W, NODE_H)

    def _colours(self):
        if self.node.is_ghost:
            return _GHOST_BG, _GHOST_BORDER
        if self.node.is_entry:
            return _ENTRY_BG, _ENTRY_BORDER
        return _PROC_BG, _PROC_BORDER

    def _tooltip(self) -> str:
        node = self.node
        if node.is_ghost:
            return "Removed stack (referenced by a later step but no longer loaded)."
        lines = [node.name]
        if node.process:
            lines.append(f"Process: {node.process}")
            if node.params:
                lines.append(_fmt_params(node.params, limit=12))
            if not (node.process in PROCESS_REGISTRY):
                lines.append("(no replay handler registered)")
        else:
            lines.append("Entry point")
        return "\n".join(lines)

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        bg, border = self._colours()
        rect = self.boundingRect().adjusted(1, 1, -1, -1)

        pen = QtGui.QPen(QtGui.QColor(border))
        pen.setWidth(3 if self.isSelected() else 2)
        if self.node.is_ghost:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        elif self.node.process and not (self.node.process in PROCESS_REGISTRY):
            pen = QtGui.QPen(QtGui.QColor(_UNREPLAYABLE_BORDER))
            pen.setWidth(3 if self.isSelected() else 2)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(bg))
        painter.drawRoundedRect(rect, 9, 9)

        fm_rect = rect.adjusted(9, 6, -9, -6)
        painter.setPen(QtGui.QColor("#20303a"))

        name_font = painter.font()
        name_font.setBold(True)
        name_font.setPointSize(9)
        painter.setFont(name_font)
        fm = QtGui.QFontMetrics(name_font)
        name = fm.elidedText(
            self.node.name, QtCore.Qt.TextElideMode.ElideRight, int(fm_rect.width())
        )
        painter.drawText(
            fm_rect,
            int(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft),
            name,
        )

        small = painter.font()
        small.setBold(False)
        small.setPointSize(8)
        painter.setFont(small)
        small_font_metrics = QtGui.QFontMetrics(small)
        if self.node.process:
            proc = small_font_metrics.elidedText(
                self.node.process,
                QtCore.Qt.TextElideMode.ElideRight,
                int(fm_rect.width()),
            )
            painter.setPen(QtGui.QColor("#33556b"))
            painter.drawText(int(fm_rect.left()), int(fm_rect.top() + 30), proc)
            params = _fmt_params(self.node.params)
            if params:
                params = small_font_metrics.elidedText(
                    params, QtCore.Qt.TextElideMode.ElideRight, int(fm_rect.width())
                )
                painter.setPen(QtGui.QColor("#77848c"))
                painter.drawText(int(fm_rect.left()), int(fm_rect.top() + 46), params)
        else:
            painter.setPen(QtGui.QColor("#4f7a4f"))
            painter.drawText(
                int(fm_rect.left()), int(fm_rect.top() + 30), "entry point"
            )


class _GraphView(QtWidgets.QGraphicsView):
    """
    Pannable canvas that renders one WorkflowGraph.
    """

    node_clicked = QtCore.pyqtSignal(object)  # WorkflowNode or None

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        # Zoom now comes from the slider rather than the pointer, so it holds
        # what is in the middle of the view rather than what is under a mouse
        # that is nowhere near the canvas.
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter
        )
        self.setBackgroundBrush(QtGui.QColor("#fbfbfb"))
        self._items: Dict[str, NodeItem] = {}

    def clear(self) -> None:
        self._scene.clear()
        self._items = {}

    def render_graph(self, graph: Optional[WorkflowGraph]) -> None:
        """Lay out and draw graph (clears the canvas when None/empty)."""
        self.clear()
        if graph is None or not graph.nodes:
            return

        # Column per depth; nodes stacked vertically within a column.
        by_depth: Dict[int, list[str]] = {}
        for nid in graph.order():
            by_depth.setdefault(graph.node_depth(nid), []).append(nid)

        for depth, nids in by_depth.items():
            for row, nid in enumerate(nids):
                item = NodeItem(graph.nodes[nid])
                item.setPos(depth * (NODE_W + COL_GAP), row * (NODE_H + ROW_GAP))
                item.setZValue(1)
                self._scene.addItem(item)
                self._items[nid] = item

        for nid, item in self._items.items():
            node = graph.nodes[nid]
            for pid in node.inputs:
                self._add_edge(pid, nid)
            for role, pid in node.aux.items():
                self._add_edge(pid, nid, role=role)

        self._scene.setSceneRect(
            self._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40)
        )

    def _add_edge(self, parent_id: str, child_id: str, role: str = "") -> None:
        """Draw a curved arrow from the parent box to the child box."""
        pit, cit = self._items.get(parent_id), self._items.get(child_id)
        if pit is None or cit is None:
            return
        p0 = pit.pos() + QtCore.QPointF(NODE_W, NODE_H / 2)
        p1 = cit.pos() + QtCore.QPointF(0, NODE_H / 2)
        dx = max(30.0, (p1.x() - p0.x()) / 2)
        path = QtGui.QPainterPath(p0)
        path.cubicTo(p0 + QtCore.QPointF(dx, 0), p1 - QtCore.QPointF(dx, 0), p1)

        colour = QtGui.QColor("#8aa0ad" if not role else "#c98a2b")
        pen = QtGui.QPen(colour, 1.6)
        if role:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        edge = self._scene.addPath(path, pen)
        edge.setZValue(0)

        self._add_arrow_head(p1, p1 - QtCore.QPointF(dx, 0), colour)
        if role:
            lbl = self._scene.addText(role)
            lbl.setDefaultTextColor(colour)
            f = lbl.font()
            f.setPointSize(7)
            lbl.setFont(f)
            mid = (p0 + p1) / 2
            lbl.setPos(mid.x() - 12, mid.y() - 16)
            lbl.setZValue(0)

    def _add_arrow_head(
        self, tip: QtCore.QPointF, from_pt: QtCore.QPointF, colour: QtGui.QColor
    ) -> None:
        angle = math.atan2(tip.y() - from_pt.y(), tip.x() - from_pt.x())
        size = 8.0
        left = tip - QtCore.QPointF(
            size * math.cos(angle - math.pi / 7), size * math.sin(angle - math.pi / 7)
        )
        right = tip - QtCore.QPointF(
            size * math.cos(angle + math.pi / 7), size * math.sin(angle + math.pi / 7)
        )
        poly = QtGui.QPolygonF([tip, left, right])
        head = self._scene.addPolygon(poly, QtGui.QPen(colour, 1), QtGui.QBrush(colour))
        head.setZValue(0)

    def set_zoom(self, percent: int) -> None:
        """
        Scale the canvas to percent of the graph's natural size.

        The transform is replaced rather than multiplied, so repeated calls do
        not compound and the view can always be put back at 100%.
        """
        factor = max(ZOOM_MIN_PCT, min(int(percent), ZOOM_MAX_PCT)) / 100.0
        self.setTransform(QtGui.QTransform().scale(factor, factor))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        item = self.itemAt(event.pos())
        while item is not None and not isinstance(item, NodeItem):
            item = item.parentItem()
        self.node_clicked.emit(item.node if isinstance(item, NodeItem) else None)
        super().mousePressEvent(event)


class WorkflowView(QtWidgets.QWidget):
    """
    Full workflow panel: disjoint workflow picker, zoom slider, graph canvas
    and node details.

    Signals
    -------
    replay_requested(object)
        Emitted with the currently shown WorkflowGraph when Replay is
        pressed.
    export_requested(object)
        Emitted with the currently shown WorkflowGraph when Export is pressed.
    import_requested()
        Emitted when Import is pressed.
    node_selected(object)
        Emitted with a WorkflowNode (or None) when the selection changes.
    """

    replay_requested = QtCore.pyqtSignal(object)
    export_requested = QtCore.pyqtSignal(object)
    import_requested = QtCore.pyqtSignal()
    node_selected = QtCore.pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._graphs: list[WorkflowGraph] = []
        self._imported: list[WorkflowGraph] = []
        self._stacks: list = []

        root = vbox(self, (6, 6, 6, 6), 6)

        bar = hbox(spacing=8)
        bar.addWidget(label("Workflow:", "font-weight: bold;"))
        self._combo = QtWidgets.QComboBox()
        self._combo.setMinimumWidth(240)
        self._combo.setToolTip("Choose which disjoint workflow to view")
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        bar.addWidget(self._combo)

        self._replay_btn = QtWidgets.QPushButton("Replay Workflow")
        self._replay_btn.setToolTip(
            "Re-run this workflow, feeding chosen stacks in at its entry points"
        )
        self._replay_btn.clicked.connect(self._emit_replay)
        bar.addWidget(self._replay_btn)

        self._export_btn = QtWidgets.QPushButton("Export...")
        self._export_btn.setToolTip(
            "Save this workflow to a file (stacks are not exported)."
        )
        self._export_btn.clicked.connect(self._emit_export)
        bar.addWidget(self._export_btn)

        self._import_btn = QtWidgets.QPushButton("Import...")
        self._import_btn.setToolTip("Open a workflow file for replay on loaded stacks")
        self._import_btn.clicked.connect(self.import_requested.emit)
        bar.addWidget(self._import_btn)

        self._close_btn = QtWidgets.QPushButton("Close")
        self._close_btn.setToolTip("Stop showing this imported workflow")
        self._close_btn.clicked.connect(self._close_imported)
        bar.addWidget(self._close_btn)
        bar.addStretch()

        bar.addWidget(label("Zoom", "color: #777; font-size: 11px;"))
        self._zoom = QLabeledSlider(QtCore.Qt.Orientation.Horizontal)
        self._zoom.setRange(ZOOM_MIN_PCT, ZOOM_MAX_PCT)
        self._zoom.setValue(ZOOM_DEFAULT_PCT)
        self._zoom.setFixedWidth(190)
        self._zoom.setToolTip("Scale of the workflow canvas, as a percentage")
        bar.addWidget(self._zoom)

        self._count_lbl = label("", "color: #777; font-size: 11px;")
        bar.addWidget(self._count_lbl)
        root.addLayout(bar)

        ##########
        # CANVAS #
        ##########

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._view = _GraphView()
        self._view.node_clicked.connect(self._on_node_clicked)
        self._zoom.valueChanged.connect(self._view.set_zoom)
        splitter.addWidget(self._view)

        self._details = QtWidgets.QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMinimumWidth(220)
        self._details.setMaximumWidth(360)
        self._details.setStyleSheet("font-size: 11px;")
        self._details.setPlainText("Select a node to see its details.")
        splitter.addWidget(self._details)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 280])
        root.addWidget(splitter, stretch=1)

        self._empty_note = label(
            "No workflow yet : load and process stacks, then return here.",
            "color: #888; font-style: italic;",
            center=True,
        )
        root.addWidget(self._empty_note)
        self._update_enabled()

    def set_stacks(self, stacks) -> None:
        """Rebuild the workflow from stacks and refresh the picker."""
        self._stacks = list(stacks)
        self._rebuild()

    def add_imported(self, graph: WorkflowGraph) -> None:
        """Show a workflow read from a file alongside the session's own, and select it."""
        self._imported.append(graph)
        self._rebuild(select=graph)

    def current_graph(self) -> Optional[WorkflowGraph]:
        """The workflow currently shown, or None."""
        i = self._combo.currentIndex()
        return self._graphs[i] if 0 <= i < len(self._graphs) else None

    def current_is_imported(self) -> bool:
        """Whether the workflow on screen came from a file rather than the store."""
        graph = self.current_graph()
        return any(graph is g for g in self._imported)

    ###########
    # PRIVATE #
    ###########

    def _rebuild(self, select: Optional[WorkflowGraph] = None) -> None:
        """
        Refill the picker from the store's workflows plus any imported ones,
        keeping the current selection where it still exists.
        """
        prev = self._combo.currentText()
        self._graphs = WorkflowGraph.from_stacks(self._stacks) + list(self._imported)

        self._combo.blockSignals(True)
        self._combo.clear()
        for i, g in enumerate(self._graphs):
            tag = "[file] " if any(g is imported for imported in self._imported) else ""
            self._combo.addItem(f"{i + 1}. {tag}{g.label()}")
        if select is not None:
            idx = next(
                (i for i, g in enumerate(self._graphs) if g is select),
                self._combo.count() - 1,
            )
        else:
            found = self._combo.findText(prev)
            idx = found if found >= 0 else 0
        self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

        self._empty_note.setVisible(not self._graphs)
        self._render_current()
        self._update_enabled()

    def _close_imported(self) -> None:
        """Drop the imported workflow on screen from the picker."""
        graph = self.current_graph()
        if graph is None:
            return
        self._imported = [g for g in self._imported if g is not graph]
        self._rebuild()

    def _render_current(self) -> None:
        graph = self.current_graph()
        self._view.render_graph(graph)
        if graph is not None:
            entries = len(graph.entry_points())
            self._count_lbl.setText(f"{graph.size()} stacks, {entries} entry point(s)")
        else:
            self._count_lbl.setText("")
        self._details.setPlainText("Select a node to see its details.")

    def _on_combo_changed(self, _i: int) -> None:
        self._render_current()
        self._update_enabled()

    def _on_node_clicked(self, node) -> None:
        self.node_selected.emit(node)
        self._details.setPlainText(self._describe(node))

    @staticmethod
    def _describe(node) -> str:
        if node is None:
            return "Select a node to see its details."
        if getattr(node, "is_ghost", False):
            return (
                f"{node.name}\n\nThis stack was referenced by a later step but is "
                "no longer loaded. Reload or re-create it to replay from here."
            )
        lines = [node.name, ""]
        if node.process:
            lines.append(f"Process: {node.process}")
            if not (node.process in PROCESS_REGISTRY):
                lines.append("⚠ No replay handler registered for this process.")
            if node.params:
                lines.append("")
                lines.append("Parameters:")
                for k, v in node.params.items():
                    lines.append(f"  • {k} = {v}")
            if node.aux:
                lines.append("")
                lines.append("Auxiliary inputs: " + ", ".join(node.aux))
        else:
            lines.append("Entry point (workflow root).")
        stack = getattr(node, "stack", None)
        if stack is not None:
            lines.append("")
            lines.append(f"Shape: {tuple(stack.data.shape)}")
        return "\n".join(lines)

    def _emit_replay(self) -> None:
        graph = self.current_graph()
        if graph is not None:
            self.replay_requested.emit(graph)

    def _emit_export(self) -> None:
        graph = self.current_graph()
        if graph is not None:
            self.export_requested.emit(graph)

    def _update_enabled(self) -> None:
        has_graph = self.current_graph() is not None
        self._replay_btn.setEnabled(has_graph)
        self._export_btn.setEnabled(has_graph)
        self._close_btn.setEnabled(self.current_is_imported())
