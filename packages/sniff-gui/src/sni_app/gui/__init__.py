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
Import with: from sni_app.gui import ...
"""

from sni_app.gui.main_gui import (
    AnalysisTab,
    MainWindow,
    ProcessingTab,
    ViewfinderTemplate,
    main,
)
from sni_app.gui.panels.io import ExportBox
from sni_app.gui.panels.logging import StatusPanel
from sni_app.gui.panels.stack_store import (
    StackItem,
    StackListPanel,
    StackStore,
)
from sni_app.gui.panels.viewfinder import ImageWorkspace
from sni_app.gui.shared import JobRunnerMixin, StackWorker
from sni_app.gui.tabs.stack_stitcher import StitcherDialog, stitch_stacks
from sni_app.gui.tabs.analysis_tab import ComputePanel
from sni_app.gui.tabs.full_processing_tab import FullProcessingTab, WorkflowView
from sni_app.gui.tabs.simulation_tab import SimulationTab
from sni_app.gui.tabs.pre_processing_tab import (
    FunctionRunner,
    PreProcessingPanel,
    RoiToolsPanel,
    StackLoader,
)

__all__ = [
    "main",
    "MainWindow",
    "ViewfinderTemplate",
    "ProcessingTab",
    "AnalysisTab",
    "PreProcessingPanel",
    "FunctionRunner",
    "StackLoader",
    "RoiToolsPanel",
    "ComputePanel",
    "SimulationTab",
    "FullProcessingTab",
    "WorkflowView",
    "StackStore",
    "StackListPanel",
    "StackItem",
    "ImageWorkspace",
    "ExportBox",
    "StatusPanel",
    "StackWorker",
    "JobRunnerMixin",
    "StitcherDialog",
    "stitch_stacks",
]
