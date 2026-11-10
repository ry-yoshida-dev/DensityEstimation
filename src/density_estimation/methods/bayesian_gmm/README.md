# bayesian_gmm

## Overview

Parameters for scikit-learn `BayesianGaussianMixture`, whose weight prior prunes unneeded components. The resulting `MixtureProcessor` supports `sample`.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `BayesianGMMParameters`. |
| [weight_prior_type.py](./weight_prior_type.py) | `BayesianGMMWeightPriorType` enum. |

## Examples

```python
from density_estimation import BayesianGMMParameters

processor = BayesianGMMParameters(num_components=10).build_processor()
processor.fit(X)
print(processor.components.get_effective_component_indices(weight_threshold=0.01))
```
