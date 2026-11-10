from __future__ import annotations

import numpy as np

from density_estimation.types import FloatArray


def standard_normal(
    num_samples: int = 200, num_features: int = 2, *, seed: int = 0
) -> FloatArray:
    """Return samples from a standard normal distribution."""
    return np.random.default_rng(seed).standard_normal((num_samples, num_features))


def two_blobs(*, seed: int = 0) -> FloatArray:
    """Return two well-separated 2-D Gaussian blobs."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((60, 2))
    b = rng.standard_normal((60, 2)) + np.array([8.0, 8.0])
    return np.vstack([a, b])
