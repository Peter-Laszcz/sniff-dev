
# SNIFF

SNIFF is a software framework developed to support the complete data handling workflow associated with spectroscopic neutron imaging. It brings together pre-experiment modelling, multidimensional image processing, wavelength-resolved analysis, and workflow automation within a single environment. The framework can be accessed interactively or via Python API. These approaches are encapsulated as two packages that share the `sni_app` namespace:

`sniff-core`
: Headless API holding all functionalities necessary for complete processing e.g. stack I/O, pre-processing, analysis, and replayable workflows.

`sniff-gui`
: Built on PyQt6 and dependent on the core library.

## Installation

If installing ``sniff-gui``, you must first install ``sniff-core`` as it is given as a core dependency.

```console
$ pip install -e packages/sniff-core
$ pip install -e packages/sniff-gui
```

Launch the desktop application with:

```console
$ sniff
```


## Documentation

```{toctree}
:maxdepth: 2

api/core
api/gui
```

```{toctree}
:hidden:

genindex
```
