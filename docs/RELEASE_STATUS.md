# Release Notes

This package contains the Fovea training engine, tests, dataset conversion scripts,
measurement tools, and baseline launchers used in the paper.

Dataset paths are configured through `FOVEA_DATA`. Datasets, trained state,
experiment logs, and the external Marius source tree are not included.

The engine uses Linux, PyTorch, and CUDA. Partitioned training uses direct I/O
and pinned host memory.
