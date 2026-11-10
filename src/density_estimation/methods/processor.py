from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import NotFittedError

from ..grid_axes import GridAxes
from ..method import DensityEstimationMethod
from ..results import DensityGrid, DensityScores
from ..types import FloatArray, NumericArray
from ..validation import as_sample_matrix


@dataclass
class DensityProcessor(ABC):
    """
    Unified interface over a fitted-on-demand density estimator.

    Subclasses provide the raw fitting and log-density computation; this base
    class validates inputs, tracks the fitted feature count, and wraps outputs
    in result types.
    """

    _num_features: int | None = field(default=None, init=False, repr=False)

    def fit(self, X: NumericArray) -> None:
        """
        Fit the density estimator.

        Parameters
        ----------
        X : NumericArray
            The training samples with shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If X is not a non-empty finite 2-D array, or violates a
            method-specific requirement such as a minimum sample count.
        """
        samples = as_sample_matrix(X)
        self._fit(samples)
        self._num_features = self._get_num_columns(samples)

    def fit_score_samples(self, X: NumericArray) -> DensityScores:
        """
        Fit the density estimator and score its training samples.

        Parameters
        ----------
        X : NumericArray
            The training samples with shape (n_samples, n_features).

        Returns
        -------
        DensityScores
            The log densities of the training samples. Methods that support it
            exclude each sample's own contribution (see the method's
            documentation).

        Raises
        ------
        ValueError
            If X is not a non-empty finite 2-D array, or violates a
            method-specific requirement such as a minimum sample count.
        """
        samples = as_sample_matrix(X)
        log_densities = self._fit_score_samples(samples)
        self._num_features = self._get_num_columns(samples)
        return DensityScores(log_densities)

    def score_samples(self, X: NumericArray) -> DensityScores:
        """
        Evaluate the fitted log density at the given samples.

        Parameters
        ----------
        X : NumericArray
            The samples with shape (n_samples, n_features).

        Returns
        -------
        DensityScores
            The log densities of the samples.

        Raises
        ------
        NotFittedError
            If the processor has not been fitted yet.
        ValueError
            If X is not a non-empty finite 2-D array with the fitted number of
            features.
        """
        return DensityScores(self._score_samples(self._as_fitted_sample_matrix(X)))

    def evaluate_grid(self, axes: GridAxes) -> DensityGrid:
        """
        Evaluate the fitted log density at every point of a grid.

        Parameters
        ----------
        axes : GridAxes
            The grid coordinates, one axis per feature.

        Returns
        -------
        DensityGrid
            The log densities on the grid.

        Raises
        ------
        NotFittedError
            If the processor has not been fitted yet.
        ValueError
            If the grid dimension differs from the fitted number of features.
        """
        if axes.num_dimensions != self.num_features:
            raise ValueError(
                f"axes must have {self.num_features} dimensions as in fit, got {axes.num_dimensions}"
            )
        return DensityGrid(axes, self.score_samples(axes.points).log_densities)

    @property
    @abstractmethod
    def method(self) -> DensityEstimationMethod:
        """
        Get the density estimation method this processor implements.

        Returns
        -------
        DensityEstimationMethod
            The method.
        """

    @property
    def is_fitted(self) -> bool:
        """
        Check whether the processor has been fitted.

        Returns
        -------
        bool
            True after a successful fit or fit_score_samples call.
        """
        return self._num_features is not None

    @property
    def num_features(self) -> int:
        """
        Get the number of features seen during fit.

        Returns
        -------
        int
            The number of features.

        Raises
        ------
        NotFittedError
            If the processor has not been fitted yet.
        """
        if self._num_features is None:
            raise NotFittedError(
                "Processor is not fitted yet. Call fit() or fit_score_samples() first."
            )
        return self._num_features

    def _as_fitted_sample_matrix(self, X: NumericArray) -> FloatArray:
        num_features = self.num_features
        samples = as_sample_matrix(X)
        if self._get_num_columns(samples) != num_features:
            raise ValueError(
                f"X must have {num_features} features as in fit, got {self._get_num_columns(samples)}"
            )
        return samples

    @staticmethod
    def _get_num_columns(samples: FloatArray) -> int:
        return np.size(samples, 1)

    @abstractmethod
    def _fit(self, samples: FloatArray) -> None: ...

    @abstractmethod
    def _score_samples(self, samples: FloatArray) -> FloatArray: ...

    def _fit_score_samples(self, samples: FloatArray) -> FloatArray:
        self._fit(samples)
        return self._score_samples(samples)
