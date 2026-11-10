from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast, override

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ..types import BoolArray, FloatArray


@dataclass(frozen=True)
class DensityScores:
    """
    Immutable log-density values of a set of samples.

    Log densities may be ``-inf`` where the estimated density is zero (e.g.,
    outside the support of a compact kernel) and ``+inf`` where it diverges
    (e.g., duplicate points under k-NN estimation), but never NaN.

    Attributes
    ----------
    log_densities : FloatArray
        The log densities as a read-only 1-D float64 array owned by this
        instance.
    """

    log_densities: FloatArray = field(init=False, repr=True)

    def __init__(self, log_densities: ArrayLike) -> None:
        owned = np.array(log_densities, dtype=np.float64).ravel()
        if owned.size == 0:
            raise ValueError("log_densities must not be empty")
        if np.any(np.isnan(owned)):
            raise ValueError("log_densities must not contain NaN")
        owned.setflags(write=False)
        object.__setattr__(self, "log_densities", owned)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DensityScores):
            return NotImplemented
        return bool(np.array_equal(self.log_densities, other.log_densities))

    @override
    def __hash__(self) -> int:
        # Adding zero maps -0.0 to 0.0, which array_equal treats as equal.
        return hash((self.log_densities + 0.0).tobytes())

    @property
    def densities(self) -> FloatArray:
        """
        Get the densities.

        Returns
        -------
        FloatArray
            The exponentiated log densities with shape (n,).
        """
        return np.exp(self.log_densities)

    @property
    def total_log_likelihood(self) -> float:
        """
        Get the log-likelihood of all samples together.

        Returns
        -------
        float
            The sum of the log densities.
        """
        return float(np.sum(self.log_densities))

    @property
    def mean_log_likelihood(self) -> float:
        """
        Get the average log-likelihood per sample.

        Returns
        -------
        float
            The mean of the log densities, comparable across sample sizes.
        """
        return float(np.mean(self.log_densities))

    @property
    def min_log_density(self) -> float:
        """
        Get the lowest log density.

        Returns
        -------
        float
            The minimum log density.
        """
        return float(np.min(self.log_densities))

    @property
    def max_log_density(self) -> float:
        """
        Get the highest log density.

        Returns
        -------
        float
            The maximum log density.
        """
        return float(np.max(self.log_densities))

    @property
    def probability_weights(self) -> FloatArray:
        """
        Get the densities normalized to sum to one over these samples.

        Computed in log space, so it stays accurate when every density is
        far below the smallest representable float.

        Returns
        -------
        FloatArray
            Non-negative weights with shape (n,) that sum to one.

        Raises
        ------
        ValueError
            If every log density is ``-inf`` or any is ``+inf``, so the weights
            are undefined.
        """
        max_log_density = self.max_log_density
        if not np.isfinite(max_log_density):
            raise ValueError(
                f"probability weights are undefined when the maximum log density is {max_log_density}"
            )
        weights: FloatArray = np.exp(self.log_densities - max_log_density)
        return weights / weights.sum()

    def get_quantile_threshold(self, quantile: float) -> float:
        """
        Get the log density below which a given fraction of samples falls.

        Parameters
        ----------
        quantile : float
            The fraction of samples in the range [0, 1].

        Returns
        -------
        float
            The log-density quantile.

        Raises
        ------
        ValueError
            If quantile is outside the range [0, 1].
        """
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        return float(np.quantile(self.log_densities, quantile))

    def get_below_threshold_mask(self, log_density_threshold: float) -> BoolArray:
        """
        Get the mask of samples whose log density is below a threshold.

        Parameters
        ----------
        log_density_threshold : float
            The log density threshold.

        Returns
        -------
        BoolArray
            The mask with shape (n,), True where the log density is strictly
            below the threshold.
        """
        return np.less(self.log_densities, log_density_threshold)

    def get_outlier_mask(self, contamination: float) -> BoolArray:
        """
        Get the mask of the lowest-density samples.

        Parameters
        ----------
        contamination : float
            The expected fraction of outliers in the range (0, 0.5].

        Returns
        -------
        BoolArray
            The mask with shape (n,), True for the ``ceil(contamination * n)``
            samples with the lowest log density. Ties are broken by the
            smaller index.

        Raises
        ------
        ValueError
            If contamination is outside the range (0, 0.5].
        """
        if not 0.0 < contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination}")
        num_outliers = math.ceil(contamination * len(self))
        mask = np.zeros(len(self), dtype=np.bool_)
        mask[self.get_lowest_indices(num_outliers)] = True
        return mask

    def get_outlier_indices(self, contamination: float) -> list[int]:
        """
        Get the indices of the lowest-density samples.

        Parameters
        ----------
        contamination : float
            The expected fraction of outliers in the range (0, 0.5].

        Returns
        -------
        list[int]
            The outlier indices in ascending order.

        Raises
        ------
        ValueError
            If contamination is outside the range (0, 0.5].
        """
        return np.flatnonzero(self.get_outlier_mask(contamination)).tolist()

    def get_highest_indices(self, count: int) -> list[int]:
        """
        Get the indices of the samples with the highest log density.

        Parameters
        ----------
        count : int
            The number of indices to return, in the range [1, n].

        Returns
        -------
        list[int]
            The indices ordered by descending log density, ties broken by the
            smaller index.

        Raises
        ------
        ValueError
            If count is outside the range [1, n].
        """
        self._validate_count(count)
        order = np.argsort(-self.log_densities, kind="stable")
        return cast(list[int], order[:count].tolist())

    def get_lowest_indices(self, count: int) -> list[int]:
        """
        Get the indices of the samples with the lowest log density.

        Parameters
        ----------
        count : int
            The number of indices to return, in the range [1, n].

        Returns
        -------
        list[int]
            The indices ordered by ascending log density, ties broken by the
            smaller index.

        Raises
        ------
        ValueError
            If count is outside the range [1, n].
        """
        self._validate_count(count)
        order = np.argsort(self.log_densities, kind="stable")
        return cast(list[int], order[:count].tolist())

    def _validate_count(self, count: int) -> None:
        if not 1 <= count <= len(self):
            raise ValueError(f"count must be in [1, {len(self)}], got {count}")

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        if dtype is None or np.dtype(dtype) == self.log_densities.dtype:
            return self.log_densities.copy() if copy else self.log_densities
        if copy is False:
            raise ValueError(
                f"cannot convert log densities to {np.dtype(dtype)} without copying"
            )
        return self.log_densities.astype(dtype)

    def __iter__(self) -> Iterator[float]:
        return iter(cast(list[float], self.log_densities.tolist()))

    def __len__(self) -> int:
        return len(self.log_densities)

    def __getitem__(self, index: int) -> float:
        """
        Get the log density at the given index.

        Parameters
        ----------
        index : int
            The index to look up.

        Returns
        -------
        float
            The log density at the given index.

        Raises
        ------
        IndexError
            If index is outside the range [0, len(self)).
        """
        if not 0 <= index < len(self):
            raise IndexError(
                f"index: {index} is out of range for {len(self)} log densities"
            )
        log_density: float = self.log_densities.item(index)
        return log_density
