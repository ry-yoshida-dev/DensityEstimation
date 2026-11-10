"""Validation of sample matrices passed to density estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray


def as_sample_matrix(samples: ArrayLike, name: str = "X") -> FloatArray:
    """
    Convert samples to a validated float64 matrix.

    Parameters
    ----------
    samples : ArrayLike
        The samples with shape (n_samples, n_features).
    name : str
        The argument name used in error messages.

    Returns
    -------
    FloatArray
        The samples as a float64 array with shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If the samples are not 2-D, are empty, or contain non-finite values.
    """
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, n_features), got {matrix.ndim}-D"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix
