# gmm

## Overview

Parameters for scikit-learn `GaussianMixture`. The resulting `MixtureProcessor` supports `sample`.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `GMMParameters`. |

## Examples

```python
from density_estimation import GMMParameters

processor = GMMParameters(num_components=3).build_processor()
processor.fit(X)
print(processor.get_information_criteria(X).bic)
```
