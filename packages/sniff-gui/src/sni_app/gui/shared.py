"""Shared GUI styles and small widget/layout factories used across every panel."""

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from superqt import QElidingLabel

# PyQt Graph config
pg.setConfigOption("background", "#2b2b2b")
pg.setConfigOption("foreground", "#dddddd")
pg.setConfigOption("imageAxisOrder", "row-major")  # array[row, col],  top-left origin

BTN_STYLE_RED = (
    "QPushButton { background-color: #cc3333; color: white; font-weight: bold;"
    " padding: 6px 14px; border-radius: 4px; border: 1px solid #992222; }"
    "QPushButton:hover { background-color: #e04444; }"
    "QPushButton:pressed { background-color: #992222; }"
)

BTN_STYLE_RUN = (
    "QPushButton { background-color: #3db83d; color: white; font-weight: bold;"
    " padding: 6px 14px; border-radius: 4px; border: 1px solid #2a8a2a; }"
    "QPushButton:hover:enabled { background-color: #4fcc4f; }"
    "QPushButton:pressed:enabled { background-color: #2a8a2a; }"
    "QPushButton:disabled { background-color: #cfcfcf; color: #909090;"  # inaccessible
    " border: 1px solid #bbbbbb; }"
)

PANEL_HEADER_STYLE = "font-weight: bold; font-size: 13px; padding: 6px 8px;"
"""Heading of a side-column panel, sitting above the panel's box."""


def popup(parent: QtWidgets.QWidget, title: str, body: str) -> None:
    """Display an information popup box."""
    dlg = QtWidgets.QMessageBox(parent)
    dlg.setWindowTitle(title)
    dlg.setText(body)
    dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
    dlg.exec()


def confirm(parent: QtWidgets.QWidget, title: str, body: str) -> bool:
    """Yes/no popup returning bool depending on response"""
    yes = QtWidgets.QMessageBox.StandardButton.Yes
    return (
        QtWidgets.QMessageBox.question(
            parent, title, body, yes | QtWidgets.QMessageBox.StandardButton.No
        )
        == yes
    )


def confirm_optout(
    parent: QtWidgets.QWidget,
    title: str,
    body: str,
    opt_out_text: str = "Don't ask again",
) -> tuple[bool, bool]:
    """
    Yes/no popup with "don't ask again" tick box.

    Returns
    -------
    tuple[bool, bool]
        (whether the user said yes, whether they ticked the box).
    """
    buttons = QtWidgets.QMessageBox.StandardButton
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(body)
    box.setIcon(QtWidgets.QMessageBox.Icon.Question)
    box.setStandardButtons(buttons.Yes | buttons.No)
    box.setDefaultButton(buttons.No)
    tick = QtWidgets.QCheckBox(opt_out_text)
    box.setCheckBox(tick)
    box.exec()
    return box.standardButton(box.clickedButton()) == buttons.Yes, tick.isChecked()


def vbox(parent=None, margins=None, spacing=None) -> QtWidgets.QVBoxLayout:
    """Return a QVBoxLayout with optional margins/spacing."""
    return _box(QtWidgets.QVBoxLayout, parent, margins, spacing)


def hbox(parent=None, margins=None, spacing=None) -> QtWidgets.QHBoxLayout:
    """Return a QHBoxLayout with optional margins/spacing."""
    return _box(QtWidgets.QHBoxLayout, parent, margins, spacing)


def _box(cls, parent, margins, spacing):
    lay = cls(parent) if parent is not None else cls()
    if margins is not None:
        lay.setContentsMargins(*margins)
    if spacing is not None:
        lay.setSpacing(spacing)
    return lay


def label(
    text: str = "", style: str = "", center: bool = False, wrap: bool = False
) -> QtWidgets.QLabel:
    """Return a QLabel with optional stylesheet, alignment, and word wrap."""
    lbl = QtWidgets.QLabel(text)
    if style:
        lbl.setStyleSheet(style)
    if center:
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    if wrap:
        lbl.setWordWrap(True)
    return lbl


def eliding_label(
    text: str = "",
    elide: QtCore.Qt.TextElideMode = QtCore.Qt.TextElideMode.ElideLeft,
    style: str = "",
) -> QElidingLabel:
    """Return a QElidingLabel that elides horizontally on text."""
    lbl = QElidingLabel(text)
    lbl.setElideMode(elide)
    lbl.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
    )
    if style:
        lbl.setStyleSheet(style)
    return lbl


def vscroll(body: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    """Wrap body in vertical scrolling area."""
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(body)
    return scroll


def flat_btn(
    text: str,
    tip: str = "",
    css: str = "QPushButton { border: none; }",
    width: int = 20,
) -> QtWidgets.QPushButton:
    """Return buttons used for edit/delete in stack row."""
    btn = QtWidgets.QPushButton(text)
    btn.setFixedWidth(width)
    btn.setFlat(True)
    btn.setStyleSheet(css)
    if tip:
        btn.setToolTip(tip)
    return btn


def browse_btn(handler, width: int = 28, tip: str = "") -> QtWidgets.QPushButton:
    """Return a file browse button with wiring."""
    btn = QtWidgets.QPushButton("📁")
    btn.setFixedWidth(width)
    if tip:
        btn.setToolTip(tip)
    btn.clicked.connect(handler)
    return btn


def run_btn(handler, text: str = "Run") -> QtWidgets.QPushButton:
    """Return a run button with wiring."""
    btn = QtWidgets.QPushButton(text)
    btn.setStyleSheet(BTN_STYLE_RUN)
    btn.clicked.connect(handler)
    return btn


def spin(value: int, lo: int, hi: int) -> QtWidgets.QSpinBox:
    """Return a QSpinBox with the given range and initial value."""
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def dspin(
    value: float, lo: float, hi: float, decimals: int = 4, step: float = 1.0
) -> QtWidgets.QDoubleSpinBox:
    """Return a QDoubleSpinBox with the given range, precision, step, and value."""
    s = QtWidgets.QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    return s


class StackWorker(QtCore.QThread):
    """
    Runs a stack job (loading, processing or saving) off the GUI thread so the
    application stays responsive.

    Signals
    -------
    progress(int, int, str)  : forwarded from the job's progress callback;
                               delivered queued on the GUI thread.
    result_ready(object)     : the job's return value, on success.
    failed(str)              : the exception message, on failure.

    QThread.finished runs after either outcome and is the place
    to re-enable controls and schedule deleteLater.
    """

    progress = QtCore.pyqtSignal(int, int, str)
    result_ready = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, job, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._job = job

    def run(self) -> None:
        """Execute the job on the worker thread and emit its outcome."""
        try:
            result = self._job(self._report)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.result_ready.emit(result)

    def _report(self, current: int, total: int, name: str = "") -> None:
        """Progress callback handed to the job; re-emits as a queued signal."""
        self.progress.emit(int(current), int(total), str(name))


class JobRunnerMixin:
    """
    Runs one StackWorker job at a time on behalf of a panel, holding
    the start/progress/failure/cleanup handling every job-running panel
    shares.  Mixed into QWidget where appropriate, i.e.:
        class StackLoader(JobRunnerMixin, QtWidgets.QWidget):
    """

    _status = None
    _worker: Optional["StackWorker"] = None

    def job_running(self) -> bool:
        """Return True while this panel's own job is in flight."""
        return self._worker is not None

    def run_job(self, job, name: str, on_result) -> None:
        """
        Run job (a callable taking a progress(current, total, name)
        callback) on a stack worker to avoid non-responsiveness.
        on_result is the GUI-thread slot receiving the job's result on success.
        """
        self._set_job_busy(True)
        if self._status is not None:
            self._status.busy(name)

        worker = StackWorker(job, parent=self)
        self._worker = worker
        worker.progress.connect(self._on_job_progress)
        worker.result_ready.connect(on_result)
        worker.failed.connect(lambda msg, n=name: self._on_job_failed(n, msg))
        worker.finished.connect(lambda n=name: self._on_job_done(n))
        worker.start()

    def _set_job_busy(self, busy: bool) -> None:
        """Gate the panel's own controls while a job runs (override)."""

    def _log(self, msg: str) -> None:
        """Log via the host's log_requested signal, else the status panel."""
        sig = getattr(self, "log_requested", None)
        if sig is not None:
            sig.emit(msg)
        elif self._status is not None:
            self._status.log(msg)

    def _on_job_progress(self, current: int, total: int, name: str) -> None:
        """Advance the status progress bar from a worker progress signal."""
        if self._status is not None and total > 0:
            self._status.set_progress(current, total, name)

    def _on_job_failed(self, name: str, message: str) -> None:
        """Report a job failure (log + popup)."""
        self._log(f"'{name}' failed: {message}")
        popup(self, "Operation Failed", f"{name} failed:\n\n{message}")

    def _on_job_done(self, name: str) -> None:
        """Job finished (success or failure) : restore controls and progress."""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._status is not None:
            self._status.done(name)
        self._set_job_busy(False)


def array_to_pixmap(arr: np.ndarray, size: int = 32) -> Optional[QtGui.QPixmap]:
    """
    Convert a 2-D numpy array into pixmap for stack preview.
    """
    try:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim > 2:  # collapse leading axes
            a = a.reshape(a.shape[-2], a.shape[-1])
        if a.ndim != 2 or a.size == 0:
            return None

        finite = np.isfinite(a)
        lo, hi = (
            (float(a[finite].min()), float(a[finite].max()))
            if finite.any()
            else (0.0, 1.0)
        )
        if hi <= lo:
            hi = lo + 1.0

        norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        img8 = np.ascontiguousarray(np.nan_to_num(norm * 255.0).astype(np.uint8))
        h, w = img8.shape
        qimg = QtGui.QImage(
            img8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8
        ).copy()
        return QtGui.QPixmap.fromImage(qimg).scaled(
            size,
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
    except Exception:
        return None
