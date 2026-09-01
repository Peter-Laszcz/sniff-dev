sniff-core API
==============

.. currentmodule:: sni_app.core

The headless library. Everything below is re-exported at the package root, so
the supported import style is::

    from sni_app.core import Stack, stack_normalisation

Most work happens at the :class:`Stack` level: a stack carries its frame data,
its per-frame metadata, and the derivation history that lets a workflow be
replayed.

Stacks
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Stack
   record_derivation

Projects
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Project
   save_project
   load_project

Workflows
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   WorkflowGraph
   WorkflowNode
   ProcessSpec
   ~components.workflow.PROCESS_REGISTRY
   get_process_spec
   replay_workflow
   active_stacks
   entry_point_uuids
   default_entry_map
   workflow_mode
   aux_roles_of
   sanitise_params
   save_workflow
   save_workflows
   load_workflow
   workflow_to_dict
   workflow_from_dict
   ~io.workflow.WORKFLOW_FORMAT

Image I/O
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_img
   get_imgs_parallel
   write_img
   ~io.image.ALLOWED_EXTENSIONS
   ~io.image.FITS_EXTENSIONS
   ~io.image.TIFF_EXTENSIONS

Stack discovery and export
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   discover_and_load
   discover_stack_dirs
   list_stack_frames
   scan_experiment_txts
   resolve_run_meta_array
   ~io.stack.RUN_META_SUFFIXES
   export_stacks
   safe_file_stem

Stack processing
----------------

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
   separate_energies
   compute_shutter_indices
   overlap_correct_array
   normalise_stack_array
   ~process.stack_processes.OVERLAP_ROLES

Image processing and registration
---------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   normalise_frame
   windowed_mean
   frame_to_2d_float32
   squeeze_to_2d
   robust_percentile_limits
   extract_features
   register_frame_to_features
   image_registration
   BlackBodyFit
   ~process.img_processes.MIN_BLACK_BODIES
   ~process.img_processes.MIN_MATCH_COUNT
   ~process.img_processes.MIN_INLIER_COUNT
   ~process.img_processes.MAX_SCALE_DEVIATION
   ~process.img_processes.MAX_ROTATION_DEG
   ~process.img_processes.MAX_TRANSLATION_FRACTION
   ~process.img_processes.REGISTRATION_MIN_SAMPLES

ROIs, attenuation and cross sections
------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   roi_to_stack
   roi_to_mask
   roi_profile
   clamp_roi_to_stack
   compute_roi_stats
   relative_attenuation
   sum_of_logs_relative_attenuation
   compute_relative_attenuation
   compute_relative_attenuation_from_stacks
   compute_relative_attenuation_from_bands
   compute_sum_of_logs_relatt_exact
   atten_coefficient
   compute_atten_coefficient_from_stacks
   compute_atten_coeff_stack
   t_cross_section
   h_cross_section
   compute_total_micro_cross_section
   compute_hydrogen_cross_section
   element_number_densities
   stack_wavelengths
   block_bin_mean
   apply_prefilter
   frame_means
   ~process.roi_processes.JANIS_CATALOGUE
   ~process.roi_processes.BARNS_PER_CM2

Simulation
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   process_compound
   energy_grid
   wavelengths
   energies
   ~process.afga.SPEC_BUILDING_BLOCKS
   ~process.afga.WAVELENGTH_MIN_A
   ~process.afga.WAVELENGTH_MAX_A

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   setup_logger
   log_dir
   default_log_file
   extract_run_stats
   frame_wavelengths
   first_shutter_count
   weighting_func
   merge_weights
   keep_dir
   keep_key_weights
   txt_timestamps
