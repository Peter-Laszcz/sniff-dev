from pathlib import Path
from typing import List, Optional

from PyQt6 import QtWidgets

from sni_app.core.io.stack import export_stacks
from sni_app.gui.panels.logging import StatusPanel
from sni_app.gui.shared import JobRunnerMixin, browse_btn, hbox, popup, run_btn


class ExportBox(JobRunnerMixin, QtWidgets.QGroupBox):
    """
    Widget for exporting selected stacks.
    Exports folder of FITS files, or 3D TIFFs.
    Option for overwriting existing images.
    """

    FORMAT_TIFF = "3D TIFF (.tif / .tiff)"
    FORMAT_FITS = "FITS files (per-frame)"

    def __init__(
        self,
        status: Optional["StatusPanel"] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__("Export Selected Stacks", parent)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")
        self._status = status
        self._stacks: List[tuple] = []  # [(name, Stack)] ticked for processing

        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(8, 6, 8, 8)
        form.setSpacing(5)

        self._count_lbl = QtWidgets.QLabel("0 stack(s) selected for export")
        self._count_lbl.setStyleSheet(
            "color: #555; font-weight: normal; font-size: 11px;"
        )
        form.addRow(self._count_lbl)

        # File format
        self._format = QtWidgets.QComboBox()
        self._format.addItems([self.FORMAT_FITS, self.FORMAT_TIFF])
        self._format.setToolTip("3D TIFF (one file) or set of per-slice FITS files.")
        form.addRow("Format", self._format)

        # Output directory + browse
        dir_w = QtWidgets.QWidget()
        dir_h = hbox(dir_w, (0, 0, 0, 0), 4)
        self._dir_field = QtWidgets.QLineEdit()
        self._dir_field.setPlaceholderText("Output folder...")
        dir_h.addWidget(self._dir_field, stretch=1)
        dir_h.addWidget(browse_btn(self._browse_dir))
        form.addRow("Folder", dir_w)

        # Filename prefix
        self._name_field = QtWidgets.QLineEdit("stack")
        self._name_field.setToolTip("Filename prefix for the exported file(s).")
        form.addRow("Filename", self._name_field)

        # Options
        self._overwrite = QtWidgets.QCheckBox("Overwrite existing files")
        form.addRow(self._overwrite)

        self._export_btn = run_btn(self._export, "Export")
        self._export_btn.setEnabled(False)
        form.addRow(self._export_btn)

    def set_stacks(self, selected_pairs: List[tuple]) -> None:
        """Set the list of stacks selected for export. Format: [(name, Stack)]"""
        self._stacks = list(selected_pairs)
        n = len(self._stacks)
        self._count_lbl.setText(f"{n} stack(s) selected for export")
        self._export_btn.setEnabled(n > 0 and not self.job_running())

    def _browse_dir(self) -> None:
        """Open a folder picker and write the chosen output directory into the field."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", ""
        )
        if path:
            self._dir_field.setText(path)

    def _set_job_busy(self, busy: bool) -> None:
        """Disable Export while this box's own job is in flight."""
        self._export_btn.setEnabled(not busy and bool(self._stacks))

    def _export(self) -> None:
        """Save selected stacks (one stack per thread). Report progress."""
        if self.job_running():
            return
        if not self._stacks:
            popup(self, "Nothing Selected", "Select stacks to export.")
            return
        base_text = self._dir_field.text().strip()
        if not base_text:
            popup(self, "No Folder", "Select a valid output folder.")
            return
        base_dir = Path(base_text)
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            popup(self, "Invalid Folder", f"Cannot use folder:\n{exc}")
            return

        # Read every widget value on the thread before starting the worker.
        stem = self._name_field.text().strip() or "stack"
        overwrite = self._overwrite.isChecked()
        format = self._format.currentText()
        pairs = list(self._stacks)
        total = len(pairs)
        ext = ".tiff" if format == self.FORMAT_TIFF else ".fits"
        op_name = f"Exporting ({format})"

        def job(progress) -> tuple:
            """Write each stack to disk, returning (written, error message list)."""
            return export_stacks(
                pairs,
                stem,
                ext,
                base_dir,
                overwrite,
                progress_callback=lambda i, n: progress(i, n, op_name),
            )

        self.run_job(
            job,
            op_name,
            lambda res: self._on_export_result(res, base_dir, format, total),
        )

    def _on_export_result(
        self, result: tuple, base_dir: Path, fmt: str, total: int
    ) -> None:
        """Report the finished export (log per-stack errors, popup a summary)."""
        written, errors = result
        if self._status is not None:
            for line in errors:
                self._status.log(f"Export of {line}")
            self._status.log(
                f"Export: wrote {written}/{total} stack(s) to '{base_dir}' as {fmt}."
            )
        popup(
            self,
            "Export Complete",
            f"Exported {written} of {total} selected stack(s) to:\n  {base_dir}\n\n",
        )
