from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ...types import FloatArray
from .covariance_type import MixtureCovarianceType


@dataclass(frozen=True, eq=False)
class MixtureComponents:
    """
    Immutable fitted parameters of a Gaussian mixture.

    Attributes
    ----------
    weights : FloatArray
        The read-only component weights with shape (n_components,).
    means : FloatArray
        The read-only component means with shape (n_components, n_features).
    covariances : FloatArray
        The read-only covariances in the layout of ``covariance_type``:
        (n_components, n_features, n_features) for FULL, (n_features,
        n_features) for TIED, (n_components, n_features) for DIAG, and
        (n_components,) for SPHERICAL.
    covariance_type : MixtureCovarianceType
        The covariance structure.
    is_converged : bool
        Whether the best fitting run converged.
    num_iterations : int
        The number of iterations of the best fitting run.
    lower_bound : float
        The lower bound on the log-likelihood (or the evidence, for Bayesian
        mixtures) reached by the best fitting run.
    """

    weights: FloatArray = field(init=False)
    means: FloatArray = field(init=False)
    covariances: FloatArray = field(init=False)
    covariance_type: MixtureCovarianceType = field(init=False)
    is_converged: bool = field(init=False)
    num_iterations: int = field(init=False)
    lower_bound: float = field(init=False)

    def __init__(
        self,
        weights: ArrayLike,
        means: ArrayLike,
        covariances: ArrayLike,
        covariance_type: MixtureCovarianceType,
        is_converged: bool,
        num_iterations: int,
        lower_bound: float,
    ) -> None:
        owned_weights = np.array(weights, dtype=np.float64)
        owned_means = np.array(means, dtype=np.float64)
        owned_covariances = np.array(covariances, dtype=np.float64)
        if owned_weights.ndim != 1 or owned_weights.size == 0:
            raise ValueError("weights must be a non-empty 1-D array")
        if owned_means.ndim != 2 or owned_means.shape[0] != owned_weights.size:
            raise ValueError(
                f"means must have shape ({owned_weights.size}, n_features), got {owned_means.shape}"
            )
        num_features = np.size(owned_means, 1)
        expected_shape = self._get_covariance_shape(
            covariance_type, owned_weights.size, num_features
        )
        if owned_covariances.shape != expected_shape:
            raise ValueError(
                f"covariances must have shape {expected_shape} for {covariance_type.value} covariance, got {owned_covariances.shape}"
            )
        for array in (owned_weights, owned_means, owned_covariances):
            array.setflags(write=False)
        object.__setattr__(self, "weights", owned_weights)
        object.__setattr__(self, "means", owned_means)
        object.__setattr__(self, "covariances", owned_covariances)
        object.__setattr__(self, "covariance_type", covariance_type)
        object.__setattr__(self, "is_converged", is_converged)
        object.__setattr__(self, "num_iterations", num_iterations)
        object.__setattr__(self, "lower_bound", lower_bound)

    @staticmethod
    def _get_covariance_shape(
        covariance_type: MixtureCovarianceType, num_components: int, num_features: int
    ) -> tuple[int, ...]:
        match covariance_type:
            case MixtureCovarianceType.FULL:
                return (num_components, num_features, num_features)
            case MixtureCovarianceType.TIED:
                return (num_features, num_features)
            case MixtureCovarianceType.DIAG:
                return (num_components, num_features)
            case MixtureCovarianceType.SPHERICAL:
                return (num_components,)

    @property
    def num_components(self) -> int:
        """
        Get the number of components.

        Returns
        -------
        int
            The number of components, including ones with negligible weight.
        """
        return int(self.weights.size)

    @property
    def num_features(self) -> int:
        """
        Get the number of features.

        Returns
        -------
        int
            The number of features.
        """
        return np.size(self.means, 1)

    @property
    def num_free_parameters(self) -> int:
        """
        Get the number of free parameters of the mixture.

        Returns
        -------
        int
            The count of covariance, mean, and independent weight parameters.
        """
        covariance_parameters = self.covariance_type.get_num_covariance_parameters(
            self.num_components, self.num_features
        )
        mean_parameters = self.num_components * self.num_features
        return covariance_parameters + mean_parameters + self.num_components - 1

    @property
    def full_covariances(self) -> FloatArray:
        """
        Get every component's covariance as a full matrix.

        Returns
        -------
        FloatArray
            The covariances with shape (n_components, n_features, n_features),
            regardless of ``covariance_type``.
        """
        identity = np.eye(self.num_features)
        match self.covariance_type:
            case MixtureCovarianceType.FULL:
                return self.covariances.copy()
            case MixtureCovarianceType.TIED:
                return np.broadcast_to(
                    self.covariances,
                    (self.num_components, self.num_features, self.num_features),
                ).copy()
            case MixtureCovarianceType.DIAG:
                return self.covariances[:, :, np.newaxis] * identity
            case MixtureCovarianceType.SPHERICAL:
                return self.covariances[:, np.newaxis, np.newaxis] * identity

    def get_effective_component_indices(self, weight_threshold: float) -> list[int]:
        """
        Get the components whose weight reaches a threshold.

        Useful for Bayesian mixtures, whose prior drives unneeded components
        toward zero weight.

        Parameters
        ----------
        weight_threshold : float
            The minimum weight in the range [0, 1].

        Returns
        -------
        list[int]
            The component indices ordered by descending weight, ties broken by
            the smaller index.

        Raises
        ------
        ValueError
            If weight_threshold is outside the range [0, 1].
        """
        if not 0.0 <= weight_threshold <= 1.0:
            raise ValueError(
                f"weight_threshold must be in [0, 1], got {weight_threshold}"
            )
        order = np.argsort(-self.weights, kind="stable")
        return cast(list[int], order[self.weights[order] >= weight_threshold].tolist())
