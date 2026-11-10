# DensityEstimation

## Overview

DensityEstimation (`density_estimation`) wraps scikit-learn density estimators, plus a from-scratch k-NN estimator, behind one API: parameter dataclasses build processors with `fit`, `fit_score_samples`, `score_samples`, and `evaluate_grid`, and methods that can generate data also offer `sample`. Outputs are immutable result types: `DensityScores` (log-likelihoods, outliers, rankings), `DensityGrid` (mode, highest density regions, marginals), and `DensitySamples`. Everything is exported from the top-level `density_estimation` package.

| Method | Parameters | `sample` | Normalized density | Extra results |
|---|---|---|---|---|
| Kernel density estimation | `KDEParameters` | Gaussian and top-hat kernels | Yes | Fitted `bandwidth` |
| Gaussian mixture | `GMMParameters` | Yes | Yes | `MixtureComponents`, `InformationCriteria` (AIC/BIC) |
| Bayesian Gaussian mixture | `BayesianGMMParameters` | Yes | Yes | `MixtureComponents` with effective components |
| k-nearest neighbor | `KNNParameters` | No | No | Leave-one-out training scores |

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np

from density_estimation import GridAxes, KDEParameters

rng = np.random.default_rng(0)
X = np.vstack([rng.standard_normal((200, 2)), rng.standard_normal((200, 2)) + 4.0])

processor = KDEParameters().build_processor()
scores = processor.fit_score_samples(X)
print(scores.mean_log_likelihood, scores.get_outlier_indices(contamination=0.01))

grid = processor.evaluate_grid(GridAxes.from_samples(X, num_points=100))
print(grid.mode, grid.get_highest_density_region_mask(0.9).sum())

samples = processor.sample(1000)
print(samples.mean)
```

Choose the number of Gaussian mixture components by BIC:

```python
from density_estimation import GMMParameters

bic_by_num_components: dict[int, float] = {}
for num_components in range(1, 6):
    processor = GMMParameters(num_components=num_components).build_processor()
    processor.fit(X)
    bic_by_num_components[num_components] = processor.get_information_criteria(X).bic
print(min(bic_by_num_components, key=bic_by_num_components.__getitem__))
```
