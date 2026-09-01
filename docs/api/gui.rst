sniff-gui API
=============

.. currentmodule:: sni_app.gui

The PyQt6 desktop application. The installed ``sniff`` console script is a thin
wrapper around :func:`main`; the classes below are documented mainly for people
extending the interface rather than for downstream use.

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
   PreProcessingPanel
   FullProcessingTab
   AnalysisTab
   SimulationTab
   WorkflowView
   ComputePanel
   RoiToolsPanel
   StackLoader
   FunctionRunner

Panels and widgets
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ViewfinderTemplate
   ImageWorkspace
   StackStore
   StackListPanel
   StackItem
   ExportBox
   StatusPanel

Stitching
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StitcherDialog
   stitch_stacks

Background work
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StackWorker
   JobRunnerMixin
