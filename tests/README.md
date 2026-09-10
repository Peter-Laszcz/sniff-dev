# Core library tests

Regression tests for `sniff-core`.

```
pip install -e "./packages/sniff-core[test]"
pytest
```

These tests are also run as a workflow on every push and pull request.

## Layout

| File | Covers                                                                                                                                      |
| --- |---------------------------------------------------------------------------------------------------------------------------------------------|
| `conftest.py` | Shared fixtures (synthetic stacks, FITS folder writing, sample-data loading)                                                                |
| `test_stack_loading.py` | `Stack.from_folder` / `from_fits_list` / `from_array`, folder discovery                                                                     |
| `test_summed_frame.py` | Summed image detection                                                                                                                      |
| `test_stack_metadata.py` | Stack identity, provenance, times of flight, analysis results                                                                               |
| `test_run_metadata.py` | I.e. shutter times, spectra, shutter count                                                                                                  |
| `test_stack_processes.py` | Averaging, summing, joining, slicing, binning, normalisation, overlap correction, stitching, black-body correction, scrubbing, registration |
| `test_roi_processes.py` | ROI cropping, profiles, statistics, cross-sections, relative attenuation                                                                    |
| `test_workflow.py` | Workflow graphs, replay, and workflow files                                                                                                 |
| `test_project_and_stack_io.py` | Stack export and project files                                                                                                              |
| `test_simulation_and_utils.py` | AFGA results, timestamps, weights, logging                                                                                                  |

## Sample data

Tests marked `requires_sample_data` use bundled sample data (downscaled) consisting of an experimental stack and
an open beam stack. They are skipped if the data is absent.
