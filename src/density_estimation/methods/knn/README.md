# knn

## Overview

From-scratch k-nearest neighbor density estimation. The estimate is not normalized, so compare its values only with each other (e.g., for outlier ranking).

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `KNNParameters`. |
| [processor.py](./processor.py) | `KNNProcessor`. |
| [metric.py](./metric.py) | `KNNMetric` enum with each metric's unit ball volume. |

## Examples

```python
from density_estimation import KNNParameters

scores = KNNParameters(num_neighbors=10).build_processor().fit_score_samples(X)
print(scores.get_lowest_indices(5))
```
