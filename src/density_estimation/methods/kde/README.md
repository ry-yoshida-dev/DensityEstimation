# kde

## Overview

Kernel density estimation with scikit-learn `KernelDensity`.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `KDEParameters`. |
| [sampling_processor.py](./sampling_processor.py) | `KDEProcessor`, which also exposes the fitted bandwidth. |
| [kernel.py](./kernel.py) | `KDEKernel` enum. |
| [bandwidth_rule.py](./bandwidth_rule.py) | `KDEBandwidthRule` enum (Scott, Silverman). |
| [algorithm.py](./algorithm.py) | `KDEAlgorithm` enum of acceleration trees. |

## Examples

```python
from density_estimation import KDEKernel, KDEParameters

processor = KDEParameters(bandwidth=0.5, kernel=KDEKernel.TOPHAT).build_processor()
processor.fit(X)
print(processor.bandwidth)
```
