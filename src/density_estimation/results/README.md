# results

## Overview

Immutable result types returned by every density processor.

## Components

| Component | Description |
|-----------|-------------|
| [scores.py](./scores.py) | `DensityScores`: log densities with likelihood summaries, outlier detection, and rankings. |
| [grid.py](./grid.py) | `DensityGrid`: log densities on a grid with mode, probability mass, highest density regions, and marginals. |
| [samples.py](./samples.py) | `DensitySamples`: samples drawn from a fitted density with summary statistics. |

## Examples

```python
scores = processor.score_samples(X)
outliers = scores.get_outlier_mask(contamination=0.05)

grid = processor.evaluate_grid(GridAxes.from_samples(X, num_points=100))
marginal_x = grid.marginalize(axis=1)
```
