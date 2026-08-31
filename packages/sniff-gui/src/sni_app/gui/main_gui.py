"""
Frontend for SNIFF. Comprised of Simulation, Processing, Analysis, and Full Processing tabs
"""

import multiprocessing
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QSplashScreen
from sni_app.core.components.stack import Stack
from sni_app.core.util.logger import log_dir
from sni_app.gui.panels.io import ExportBox
from sni_app.gui.panels.logging import StatusPanel
from sni_app.gui.panels.stack_store import StackListPanel, StackStore
from sni_app.gui.panels.viewfinder import ImageWorkspace
from sni_app.gui.shared import hbox, popup, vbox
from sni_app.gui.tabs.analysis_tab import ComputePanel
from sni_app.gui.tabs.full_processing_tab import FullProcessingTab
from sni_app.gui.tabs.simulation_tab import SimulationTab
from sni_app.gui.tabs.pre_processing_tab import (
    PreProcessingPanel,
    RoiToolsPanel,
    StackLoader,
)


################
#     TABS     #
################


class ViewfinderTemplate(QtWidgets.QWidget):
    """
    Template for tabs using the stack viewfinder/ ROI plotter. Provides space for a left and right panel.
    Maintains stack list and viewfinder states between children.

    Signals
    -------
    export_requested : request for export box.
    """

    export_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        store: StackStore,
        status: StatusPanel,
        workspace: ImageWorkspace,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._status = status
        self._workspace = workspace
        self._center = QtWidgets.QWidget()  # slot the previewer moves into
        self._center_layout = vbox(self._center, (0, 0, 0, 0), 0)

    def adopt_workspace(self) -> None:
        """Move the shared previewer into this tab (call when the tab is shown)."""
        if self._workspace.parent() is not self._center:
            self._center_layout.addWidget(self._workspace)

    def _assemble(self, left: QtWidgets.QWidget, right: QtWidgets.QWidget) -> None:
        """Place left/right tabs and the previewer slot into the tab's splitter."""
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for i, (widget, stretch) in enumerate(
            ((left, 0), (self._center, 1), (right, 0))
        ):
            splitter.addWidget(widget)
            splitter.setStretchFactor(i, stretch)
        splitter.setSizes([320, 820, 340])  # section proportions
        hbox(self, (0, 0, 0, 0), 0).addWidget(splitter)

    def _side_column(self) -> QtWidgets.QWidget:
        """Return an empty column widget."""
        col = QtWidgets.QWidget()
        col.setMinimumWidth(300)
        col.setMaximumWidth(420)
        vbox(col, (0, 0, 0, 0), 0)
        return col

    def _on_stack_selected(self, name: str, stack: Stack) -> None:
        """load clicked stack into the previewer."""
        try:
            self._workspace.load_stack(stack, name)
        except Exception as exc:
            self._status.log(f"Failed to preview '{name}': {exc}")
            return
        self._status.log(f"Previewing stack '{name}'  (shape {stack.data.shape}).")


class ProcessingTab(ViewfinderTemplate):
    """
    Tab for loading and pre-processing stacks.
    Consists of the process widget on the left and the ROI box and stack box on
    the right. Loading is driven from the File menu.
    Stack box is shared with Analysis tab.
    """

    def __init__(
        self,
        store: StackStore,
        status: StatusPanel,
        workspace: ImageWorkspace,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(store, status, workspace, parent)

        # Driven from the File menu, so it carries no UI: parented here (and
        # hidden) only so its dialogs centre on the main window.
        self._loader = StackLoader(store, status, self)
        self._loader.hide()
        self._loader.log_requested.connect(status.log)

        self._pre_processing = PreProcessingPanel(store, status)
        self._pre_processing.log_requested.connect(status.log)

        self._list = StackListPanel(store, status)
        self._list.stack_selected.connect(self._on_stack_selected)
        self._list.log_requested.connect(status.log)
        self._list.export_selected_requested.connect(self.export_requested)

        left = self._side_column()
        left.layout().addWidget(self._pre_processing, stretch=1)

        self._roi_tools = RoiToolsPanel(store, workspace, status)
        self._roi_tools.log_requested.connect(status.log)

        right = self._side_column()
        right.layout().addWidget(self._roi_tools)
        right.layout().addWidget(self._list, stretch=1)

        self._assemble(left, right)

    #####################
    # FILE MENU ACTIONS #
    #####################

    def load_stacks_dialog(self) -> None:
        """Prompt for a source directory and load the stacks under it."""
        self._loader.browse_and_load()

    #####################
    # GUI STATE HELPERS #
    #####################

    def gui_state(self) -> dict:
        """Return the GUI field values for project save."""
        return {
            "source_directory": self._loader.source_directory(),
            "current_function": self._pre_processing.current_function(),
            "saved_rois": self._roi_tools.saved_rois(),
        }

    def apply_ui_state(self, state: dict) -> None:
        """Restore the field values saved by gui_state method."""
        self._loader.set_source_directory(state.get("source_directory", ""))
        fn = state.get("current_function")
        if fn:
            self._pre_processing.set_function(str(fn))
        self._roi_tools.set_saved_rois(state.get("saved_rois", {}))


class AnalysisTab(ViewfinderTemplate):
    """
    Tab for attenuation and cross-section calculation.
    Shares stack box with Processing tab.
    """

    def __init__(
        self,
        store: StackStore,
        status: StatusPanel,
        workspace: ImageWorkspace,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(store, status, workspace, parent)

        # Stack box
        self._list = StackListPanel(store, status)
        self._list.stack_selected.connect(self._on_stack_selected)
        self._list.log_requested.connect(status.log)
        self._list.export_selected_requested.connect(self.export_requested)

        right = self._side_column()  # owns the shared side-column widths
        right.layout().addWidget(self._list, stretch=1)

        # Compute panel
        self._compute = ComputePanel(status)
        self._compute.stack_produced.connect(self._on_stack_produced)

        # Sync compute panel parameters to stack list
        store.stacks_changed.connect(self._sync_compute)
        self._sync_compute()

        self._assemble(self._compute, right)

    def _sync_compute(self) -> None:
        """Push the current stack list to the compute panel."""
        self._compute.set_all_stacks(self._store.pairs())

    def _on_stack_produced(self, name: str, stack: Stack) -> None:
        """A compute dropdown produced a result : add it to the Stacks list."""
        self._store.add(stack, name=name)


##################################
#           MAIN WINDOW          #
##################################


class MainWindow(QtWidgets.QMainWindow):
    """
    Top-level window. Hosts tabs, progress bar, and menu ribbon.
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("SNIFF")
        self.resize(1440, 920)
        self.setMinimumSize(1280, 720)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = vbox(central, (0, 0, 0, 0), 0)

        # shared components
        self._status = StatusPanel()
        self._store = StackStore(self)
        self._workspace = ImageWorkspace()

        self._open_projects: list = []

        # One export box for the whole window: both stacks boxes and the File
        # menu open the same one, so its settings are shared between them.
        self._export = ExportBox(self._status)
        self._export_dialog: Optional[QtWidgets.QDialog] = None
        self._store.stacks_changed.connect(self._sync_export)
        self._store.selection_changed.connect(self._sync_export)
        self._sync_export()

        ##########
        #  Tabs  #
        ##########

        self._tab_widget = QtWidgets.QTabWidget()
        self._tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.setStyleSheet(self._build_tab_style())

        self._processing = ProcessingTab(self._store, self._status, self._workspace)
        self._tab_widget.addTab(self._processing, "Processing")
        self._analysis = AnalysisTab(self._store, self._status, self._workspace)
        self._tab_widget.addTab(self._analysis, "Analysis")
        self._full_processing = FullProcessingTab(self._store, self._status)
        self._tab_widget.addTab(self._full_processing, "Full Processing")
        self._simulation = SimulationTab(self._status)
        self._tab_widget.addTab(self._simulation, "Simulation")

        for tab in (self._processing, self._analysis):
            tab.export_requested.connect(self.export_stacks_dialog)

        self._tab_widget.currentChanged.connect(self._on_tab_changed)
        self._processing.adopt_workspace()  # Starting tab set to Processing

        root.addWidget(self._tab_widget, stretch=1)
        root.addWidget(self._status)  # status panel pinned below the tabs

        self._build_menu()

    ##################################
    #             EXPORT             #
    ##################################

    def _sync_export(self) -> None:
        """Push the stacks ticked for processing to the export box."""
        self._export.set_stacks(self._store.selected_pairs())

    def export_stacks_dialog(self) -> None:
        """
        Show the export box in its own window.
        """
        if self._export_dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Export Selected Stacks")
            vbox(dialog, (8, 8, 8, 8), 0).addWidget(self._export)
            self._export_dialog = dialog
        self._export_dialog.show()
        self._export_dialog.raise_()
        self._export_dialog.activateWindow()

    def _on_tab_changed(self, index: int) -> None:
        """Hand the shared previewer to the newly shown tab and log the switch."""
        tab = self._tab_widget.widget(index)
        if isinstance(tab, ViewfinderTemplate):
            tab.adopt_workspace()
        self._status.log(f"Switched to tab: {self._tab_widget.tabText(index).strip()}")

    @staticmethod
    def _build_tab_style() -> str:
        """Return the QSS stylesheet for tabs."""
        return """
        QTabBar::tab {
            padding: 8px 22px;
            min-width: 148px;
            font-size: 13px;
            font-weight: bold;
            border: 1px solid #aaa;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 3px;
            background-color: #d8d8d8;
        }
        QTabBar::tab:selected {
            background-color: #c0dff0;
            border-bottom: 2px solid white;
            color: #003366;
        }
        QTabBar::tab:hover:!selected {
            background-color: #e8e8e8;
        }
        QTabWidget::pane {
            border: 1px solid #aaa;
        }
        """

    #####################################
    #            Menu Ribbon            #
    #####################################

    @staticmethod
    def _add_action(
        menu: QtWidgets.QMenu, text: str, slot, shortcut: str = ""
    ) -> QtGui.QAction:
        """Append an action to the menu."""
        act = menu.addAction(text)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        return act

    def _build_menu(self) -> None:
        """Build the menu ribbon."""
        mb = self.menuBar()
        file_menu = mb.addMenu("File")
        self._add_action(file_menu, "New Project", self._new_project, "Ctrl+N")
        self._add_action(file_menu, "Open Project...", self._open_project, "Ctrl+O")
        self._add_action(file_menu, "Save Project...", self._save_project, "Ctrl+S")
        file_menu.addSeparator()
        self._load_action = self._add_action(
            file_menu,
            "Load Stacks Directory...",
            self._processing.load_stacks_dialog,
            "Ctrl+L",
        )
        self._add_action(
            file_menu,
            "Export Selected Stacks...",
            self.export_stacks_dialog,
            "Ctrl+E",
        )
        # Loading writes into the shared store, so it waits for any running job.
        self._store.busy_changed.connect(
            lambda busy: self._load_action.setEnabled(not busy)
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu, "Import Workflow...", self._import_workflow, "Ctrl+Shift+O"
        )
        self._add_action(
            file_menu, "Export Workflow...", self._export_workflow, "Ctrl+Shift+E"
        )
        file_menu.addSeparator()
        self._add_action(file_menu, "Quit", self.close, "Ctrl+Q")

        help_menu = mb.addMenu("Help")
        self._add_action(help_menu, "Show Log", self._show_log, "Ctrl+Shift+L")
        self._add_action(help_menu, "Browse Log Folder", self._browse_log_folder)
        help_menu.addSeparator()
        self._add_action(help_menu, "About", self._about)

    def _show_log(self) -> None:
        """Open the session log in its own window."""
        self._status.show_log_window(self)

    def _show_full_processing(self) -> None:
        """Bring the Full Processing tab forward : where workflows live."""
        self._tab_widget.setCurrentWidget(self._full_processing)

    def _import_workflow(self) -> None:
        """Read a workflow file into the Full Processing tab."""
        self._show_full_processing()
        self._full_processing.import_workflow()

    def _export_workflow(self) -> None:
        """Write the workflow shown in the Full Processing tab to a file."""
        self._show_full_processing()
        self._full_processing.export_workflow()

    def _browse_log_folder(self) -> None:
        """Open the folder SNIFF writes its logs to in the system file browser."""
        folder = log_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            popup(self, "Log Folder", f"Could not create {folder}:\n\n{exc}")
            return
        url = QtCore.QUrl.fromLocalFile(str(folder))
        if not QtGui.QDesktopServices.openUrl(url):  # no file browser
            popup(self, "Log Folder", f"Could not open a file browser.\n\n{folder}")
            return
        self._status.log(f"Opened log folder: {folder}")

    def _about(self) -> None:
        """Show the "About SNIFF" dialogue."""
        QtWidgets.QMessageBox.about(
            self,
            "About – SNIFF",
            "<b>SNI Full-Processing Framework,</b><br><br>"
            "ISIS Neutron and Muon Source, STFC 2026.<br>"
            "Source Code by Peter Laszcz, Scott Young, and Eric Ricardo Carreon Ruiz.<br>"
            "Elements used from NEAT, AFGA, and Mantid Imaging.<br>"
            "Logo by Ander Hatton.<br>",
        )

    ##################################
    #         Saving/Loading         #
    ##################################

    def _new_project(self) -> None:
        """Initialise workspace."""
        self._load_state([], {})
        self._status.log("New project : workspace cleared.")

    def _open_project(self) -> None:
        """Prompt for a project file path which is loaded."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            f"SNIFF Project (*.sniff);;All files (*)",
        )
        if path:
            self._project_io(
                "Open Project", self._load_project_file, path, "Opened project"
            )

    def _save_project(self) -> None:
        """Prompt for project file destination and save."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            f"SNIFF Project (*.sniff);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".sniff"):
            path += ".sniff"
        self._project_io("Save Project", self._save_project_file, path, "Saved project")

    def _save_project_file(self, path: str) -> None:
        """Write the current workspace to a project file at specified path."""
        from sni_app.core.io.project import save_project

        save_project(path, self._store.stacks(), self._processing.gui_state())

    def _load_project_file(self, path: str) -> None:
        """Load a project file and repopulate the workspace."""
        from sni_app.core.io.project import load_project

        loaded = load_project(path, lazy=True)
        self._load_state(
            loaded.stacks, loaded.gui_state
        )  # Keep open for stack accessibility
        self._open_projects.append(loaded)

    def _load_state(self, stacks: list, ui_state: Optional[dict] = None) -> None:
        """
        Restore workspace from stacks and UI state dict.
        """
        self._store.replace_all(stacks)
        self._processing.apply_ui_state(ui_state or {})

    def _project_io(self, title: str, action, path: str, done: str) -> None:
        """Run a project load/save and report the outcome."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            action(path)
        except Exception as exc:
            popup(self, f"{title} Failed", f"Could not {title.lower()}:\n\n{exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._status.log(f"{done}: {path}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Release all project files."""
        for lp in self._open_projects:
            try:
                lp.close()
            except Exception:
                pass
        self._open_projects = []
        super().closeEvent(event)


def _asset(name: str) -> Optional[Path]:
    """
    Return the path of a packaged asset, or None when it is not there.

    Resolved through the package rather than the current working directory, so
    the splash and icon are found however the app was launched and wherever it
    is installed.
    """
    try:
        path = resources.files("sni_app.gui") / name
        with resources.as_file(path) as real:
            return real if real.is_file() else None
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None


def main() -> None:
    # Frozen builds spawn workers by re-running this exe; without this each
    # worker would relaunch the whole GUI (see core.process.stack_processes).
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SNIFF")
    app.setStyle("Fusion")

    logo = _asset("logo.png")
    splash = QSplashScreen(QPixmap(str(logo))) if logo else QSplashScreen()
    splash.show()

    window = MainWindow()
    icon = _asset("icon.png")
    if icon is not None:
        window.setWindowIcon(QIcon(str(icon)))
    else:
        window._status.log("Window icon not found in the installed package.")

    splash.finish(window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
