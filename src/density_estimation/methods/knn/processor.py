from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import override

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors

from ...method import DensityEstimationMethod
from ...types import FloatArray
from ..processor import DensityProcessor
from .metric import KNNMetric


@dataclass
class KNNProcessor(DensityProcessor):
    """
    k-nearest neighbor density estimation.

    Estimates ``p(x) = k / (n * V * r_k(x) ** d)``, where ``r_k(x)`` is the
    distance from x to its k-th nearest training sample, V the volume of the
    metric's unit ball, and d the number of features. The estimate adapts to
    local density without a bandwidth, but it is not normalized (its integral
    diverges), so compare its values only with each other.

    fit_score_samples scores each training sample against the other n - 1
    samples, excluding the sample itself. A query point that coincides with k
    or more training samples gets a log density of ``+inf``.

    Attributes
    ----------
    num_neighbors : int
        The neighbor rank k whose distance sets the density.
    metric : KNNMetric
        The distance metric.
    """

    num_neighbors: int
    metric: KNNMetric
    _neighbors: NearestNeighbors | None = field(default=None, init=False, repr=False)
    _num_training_samples: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_neighbors <= 0:
            raise ValueError("num_neighbors must be greater than 0")

    @property
    @override
    def method(self) -> DensityEstimationMethod:
        return DensityEstimationMethod.KNN

    @override
    def _fit(self, samples: FloatArray) -> None:
        num_samples = len(samples)
        if num_samples < self.num_neighbors:
            raise ValueError(
                f"X must have at least num_neighbors={self.num_neighbors} samples, got {num_samples}"
            )
        self._neighbors = NearestNeighbors(
            n_neighbors=self.num_neighbors, metric=self.metric.value
        ).fit(samples)
        self._num_training_samples = num_samples

    @override
    def _score_samples(self, samples: FloatArray) -> FloatArray:
        distances, _ = self._get_neighbors().kneighbors(samples)
        return self._to_log_densities(
            distances[:, -1], self._num_training_samples, self._get_num_columns(samples)
        )

    @override
    def _fit_score_samples(self, samples: FloatArray) -> FloatArray:
        num_samples = len(samples)
        if num_samples <= self.num_neighbors:
            raise ValueError(
                f"X must have more than num_neighbors={self.num_neighbors} samples to exclude each sample from its own neighbors, got {num_samples}"
            )
        self._fit(samples)
        # Querying without X makes scikit-learn exclude each sample itself.
        distances, _ = self._get_neighbors().kneighbors()
        return self._to_log_densities(
            distances[:, -1], num_samples - 1, self._get_num_columns(samples)
        )

    def _get_neighbors(self) -> NearestNeighbors:
        if self._neighbors is None:
            raise NotFittedError(
                "Processor is not fitted yet. Call fit() or fit_score_samples() first."
            )
        return self._neighbors

    def _to_log_densities(
        self, kth_distances: FloatArray, num_reference_samples: int, num_features: int
    ) -> FloatArray:
        log_normalizer = (
            math.log(self.num_neighbors)
            - math.log(num_reference_samples)
            - self.metric.get_log_unit_ball_volume(num_features)
        )
        with np.errstate(divide="ignore"):
            log_distances = np.log(kth_distances)
        return log_normalizer - num_features * log_distances
