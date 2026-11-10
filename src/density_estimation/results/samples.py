from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ..types import FloatArray
from ..validation import as_sample_matrix


@dataclass(frozen=True, eq=False)
class DensitySamples:
    """
    Immutable samples drawn from a fitted density.

    Attributes
    ----------
    samples : FloatArray
        The samples as a read-only float64 array with shape
        (n_samples, n_features) owned by this instance.
    """

    samples: FloatArray = field(init=False, repr=True)

    def __init__(self, samples: ArrayLike) -> None:
        owned = as_sample_matrix(samples, name="samples").copy()
        owned.setflags(write=False)
        object.__setattr__(self, "samples", owned)

    @property
    def num_samples(self) -> int:
        """
        Get the number of samples.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.samples)

    @property
    def num_features(self) -> int:
        """
        Get the number of features of each sample.

        Returns
        -------
        int
            The number of features.
        """
        return np.size(self.samples, 1)

    @property
    def mean(self) -> FloatArray:
        """
        Get the sample mean.

        Returns
        -------
        FloatArray
            The mean with shape (n_features,).
        """
        return np.mean(self.samples, axis=0)

    @property
    def covariance(self) -> FloatArray:
        """
        Get the unbiased sample covariance.

        Returns
        -------
        FloatArray
            The covariance with shape (n_features, n_features).

        Raises
        ------
        ValueError
            If there are fewer than two samples.
        """
        if self.num_samples < 2:
            raise ValueError(
                f"covariance requires at least 2 samples, got {self.num_samples}"
            )
        return np.atleast_2d(np.cov(self.samples, rowvar=False))

    @property
    def lower_bounds(self) -> FloatArray:
        """
        Get the per-feature minimum.

        Returns
        -------
        FloatArray
            The minimum of each feature with shape (n_features,).
        """
        return np.min(self.samples, axis=0)

    @property
    def upper_bounds(self) -> FloatArray:
        """
        Get the per-feature maximum.

        Returns
        -------
        FloatArray
            The maximum of each feature with shape (n_features,).
        """
        return np.max(self.samples, axis=0)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        if dtype is None or np.dtype(dtype) == self.samples.dtype:
            return self.samples.copy() if copy else self.samples
        if copy is False:
            raise ValueError(
                f"cannot convert samples to {np.dtype(dtype)} without copying"
            )
        return self.samples.astype(dtype)

    def __len__(self) -> int:
        return self.num_samples
