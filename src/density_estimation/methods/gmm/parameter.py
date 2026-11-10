from dataclasses import dataclass
from typing import override

from sklearn.mixture import GaussianMixture

from ..mixture import MixtureCovarianceType, MixtureProcessor
from ..parameter import DensityParameters


@dataclass(frozen=True)
class GMMParameters(DensityParameters):
    """
    Parameters for scikit-learn's GaussianMixture fitted by
    expectation-maximization.

    Attributes
    ----------
    num_components : int
        The number of mixture components.
    covariance_type : MixtureCovarianceType
        The covariance structure of the components.
    convergence_tolerance : float
        The convergence threshold on the lower bound gain.
    covariance_regularization : float
        The non-negative regularization added to the covariance diagonals.
    max_iterations : int
        The maximum number of iterations per initialization.
    num_initializations : int
        The number of initializations; the best one is kept.
    random_state : int | None
        The seed for initialization and sampling.
    """

    num_components: int = 2
    covariance_type: MixtureCovarianceType = MixtureCovarianceType.FULL
    convergence_tolerance: float = 1e-3
    covariance_regularization: float = 1e-6
    max_iterations: int = 100
    num_initializations: int = 1
    random_state: int | None = 0

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.num_components <= 0:
            raise ValueError("num_components must be greater than 0")
        if self.convergence_tolerance < 0:
            raise ValueError("convergence_tolerance must be non-negative")
        if self.covariance_regularization < 0:
            raise ValueError("covariance_regularization must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if self.num_initializations <= 0:
            raise ValueError("num_initializations must be greater than 0")

    @override
    def build_processor(self) -> MixtureProcessor:
        estimator = GaussianMixture(
            n_components=self.num_components,
            covariance_type=self.covariance_type.value,
            tol=self.convergence_tolerance,
            reg_covar=self.covariance_regularization,
            max_iter=self.max_iterations,
            n_init=self.num_initializations,
            random_state=self.random_state,
        )
        return MixtureProcessor(
            estimator=estimator,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
