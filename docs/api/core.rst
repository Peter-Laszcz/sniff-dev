sniff-core API
==============

.. currentmodule:: sni_app.core

The headless library. Supported import style is::

    from sni_app.core import Stack, stack_normalisation

A user interfaces with experiment data at the :class:`Stack` level. This class carries acquisition frames,
frame metadata, experiment metadata, and processing history for replaying workflows.
Names not listed here are internal helpers.


Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Stack
   Project
   WorkflowGraph
   WorkflowNode
   ProcessSpec

Preprocessing
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   stack_overlap_correction
   stack_normalisation
   stack_scrubbing
   stack_sbkg_correction
   stack_referencing
   stack_registration
   stack_slice_acquisitions
   stack_bin_frames
   stack_avg
   stack_sum
   stack_join
   stack_stitching

Workflows
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_process_spec
   replay_workflow
   save_workflow
   save_workflows
   load_workflow
   active_stacks
   entry_point_uuids
   default_entry_map
   workflow_mode
   aux_roles_of
   sanitise_params
   record_derivation

I/O: Project
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   save_project
   load_project

I/O: Stack
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   discover_and_load
   discover_stack_dirs
   list_stack_frames
   scan_experiment_txts
   resolve_run_meta_array
   compute_shutter_indices
   export_stacks

ROI manipulation
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   roi_to_stack
   roi_to_mask
   roi_profile
   clamp_roi_to_stack
   compute_roi_stats

Analysis
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   relative_attenuation
   sum_of_logs_relative_attenuation
   atten_coefficient
   t_cross_section
   h_cross_section
   stack_wavelengths

Simulation
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   process_compound
   energy_grid
   wavelengths
   energies

Logging
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   setup_logger
   log_dir
   default_log_file

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   extract_run_stats
   frame_wavelengths
   txt_timestamps

Data Constants
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~process.roi_processes.JANIS_CATALOGUE
   ~process.roi_processes.BARNS_PER_CM2
   ~process.afga.SPEC_BUILDING_BLOCKS
   ~io.image.ALLOWED_EXTENSIONS
   ~components.workflow.PROCESS_REGISTRY

Simulation range constants
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~process.afga.WAVELENGTH_MIN_A
   ~process.afga.WAVELENGTH_MAX_A

Image processing constraint constants
-------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~process.img_processes.MIN_BLACK_BODIES
   ~process.img_processes.MIN_MATCH_COUNT
   ~process.img_processes.MIN_INLIER_COUNT
   ~process.img_processes.MAX_SCALE_DEVIATION
   ~process.img_processes.MAX_ROTATION_DEG
   ~process.img_processes.MAX_TRANSLATION_FRACTION
   ~process.img_processes.REGISTRATION_MIN_SAMPLES
