# sniff-core

The headless core library for SNIFF, a toolkit for processing and analysing
energy-resolved neutron imaging data.

`sniff-core` provides stack I/O, image and stack processing, ROI analysis,
cross-section simulation, and replayable processing workflows. It has no GUI
dependencies — the desktop application lives in the separate `sniff-gui`
package.

## Installation

```console
pip install -e .
```

## Usage

The public API is re-exported at the package root and defined by `__all__`:

```python
from pathlib import Path

from sni_app.core import discover_and_load, stack_normalisation

stacks, weights = discover_and_load(Path("data/experiment"))
```

Most work happens at the `Stack` level. A stack carries its frame data,
per-frame headers, whole-stack metadata, and the derivation history that lets a
sequence of processes be replayed against new inputs.

## Documentation

Full API documentation: https://peter-laszcz.github.io/sniff-dev/
