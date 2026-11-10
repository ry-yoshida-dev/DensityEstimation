# density_estimation

## Overview

Unified density estimation interface: parameter dataclasses build processors that fit a density, score samples, evaluate grids, and, for generative methods, draw samples.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Shared type aliases (`NumericArray`, `FloatArray`, `IntegerArray`, `BoolArray`). |
| [method.py](./method.py) | `DensityEstimationMethod` enum of available methods. |
| [validation.py](./validation.py) | Validation of sample matrices. |
| [grid_axes.py](./grid_axes.py) | `GridAxes`, the coordinates of an evaluation grid. |
| [results/](./results/README.md) | Result types shared by every method. |
| [methods/](./methods/README.md) | Parameter and processor base classes and their per-method implementations. |
