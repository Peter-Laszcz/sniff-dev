from typing import Optional

from PyQt6 import QtWidgets

from sni_app.gui.shared import hbox, vbox

LOG_AREA_STYLE = (
    "QPlainTextEdit { background: #f7f7f7; font-family: monospace;"
    " font-size: 11px; border: none; }"
)


class LogWindow(QtWidgets.QDialog):
    """
    Free-standing window showing the session log.
    """

    def __init__(
        self,
        log_area: QtWidgets.QPlainTextEdit,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SNIFF Log")
        self.resize(760, 420)

        self._log_area = log_area

        layout = vbox(self, (8, 8, 8, 8), 6)
        layout.addWidget(log_area, stretch=1)

        buttons = hbox(spacing=6)
        buttons.addStretch()
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setToolTip("Empty the log")
        clear_btn.clicked.connect(log_area.clear)
        buttons.addWidget(clear_btn)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)


class StatusPanel(QtWidgets.QWidget):
    """
    Progress bar and process readout, pinned to the bottom of the window.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.setMaximumHeight(48)
        self.setStyleSheet(
            "QWidget { background-color: #e4e4e4; border-top: 2px solid #aaa; }"
        )

        layout = vbox(self, (8, 6, 8, 6), 4)


        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setStyleSheet(LOG_AREA_STYLE)
        self._log_window: Optional[LogWindow] = None

        # Initialise progress fields.
        prog_row = hbox()

        self._proc_lbl = QtWidgets.QLabel("Ready")
        self._proc_lbl.setFixedWidth(320)
        prog_row.addWidget(self._proc_lbl)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setStyleSheet(
            "QProgressBar { border: 1px solid #aaa; border-radius: 3px;"
            " text-align: center; height: 16px; }"
            "QProgressBar::chunk { background-color: #3db83d; border-radius: 2px; }"
        )
        prog_row.addWidget(self._bar)

        self._pct_lbl = QtWidgets.QLabel("0%")
        self._pct_lbl.setFixedWidth(110)
        prog_row.addWidget(self._pct_lbl)

        layout.addLayout(prog_row)

        self.log("Ready.")
        self.debug("GUI Initialised.")

    def log(self, msg: str) -> None:
        """Append a message to the log box."""
        self._log_area.appendPlainText(msg)

    def debug(self, msg: str) -> None:
        """Append a debug message to the log box."""
        self.log(f"[DEBUG] {msg}")

    def show_log_window(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """
        Show the log in a separate window.
        """
        if self._log_window is None:
            self._log_window = LogWindow(self._log_area, parent or self.window())
        self._log_window.show()
        self._log_window.raise_()
        self._log_window.activateWindow()

    def set_progress(self, current: int, total: int, name: str = "") -> None:
        """Update the progress bar and labels ."""
        self._bar.setRange(0, total)
        self._bar.setValue(current)
        pct = int(100 * current / total) if total > 0 else 0
        self._pct_lbl.setText(f"{pct}% [{current}/{total}]")
        if name:
            self._proc_lbl.setText(f"Current Process: {name}")

    def busy(self, name: str = "") -> None:
        """Nonspecific progress animation"""
        if name:
            self._proc_lbl.setText(f"Current Process: {name}")
        self._bar.setRange(0, 0)  # moving green bar thing
        self._pct_lbl.setText("...")
        QtWidgets.QApplication.processEvents()

    def done(self, name: str = "") -> None:
        """Return the bar to 100% after operation."""
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._pct_lbl.setText("100%")
        if name:
            self._proc_lbl.setText(f"Current Process: {name}: done")
        QtWidgets.QApplication.processEvents()
