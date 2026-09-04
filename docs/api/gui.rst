sniff-gui API
=============

.. currentmodule:: sni_app.gui

SNIFF's PyQt6 GUI application. The installed ``sniff`` console script wraps :func:`main`.
The primary use for the GUI is entirely graphical interaction; the classes below are documented mainly for developers
wishing to extend the interface, rather than for everyday use.

Entry point
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   main
   MainWindow

Tabs
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ProcessingTab
   FullProcessingTab
   AnalysisTab
   SimulationTab

Panels
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StackListPanel
   StatusPanel
   ComputePanel
   RoiToolsPanel
   PreProcessingPanel

Widgets
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StackLoader
   StackStore
   StackItem
   ExportBox
   ViewfinderTemplate
   ImageWorkspace
   WorkflowView
   FunctionRunner

Stitching
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StitcherDialog
   stitch_stacks

Workers
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StackWorker
   JobRunnerMixin
