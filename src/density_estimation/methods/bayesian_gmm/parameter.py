from dataclasses import dataclass
from typing import override

from sklearn.mixture import BayesianGaussianMixture

from ..mixture import MixtureCovarianceType, MixtureProcessor
from ..parameter import DensityParameters
from .weight_prior_type import BayesianGMMWeightPriorType


@dataclass(frozen=True)
class BayesianGMMParameters(DensityParameters):
    """
    Parameters for scikit-learn's BayesianGaussianMixture fitted by
    variational inference.

    num_components is an upper bound: the weight prior drives unneeded
    components toward zero weight, which MixtureComponents exposes through
    get_effective_component_indices.

    Attributes
    ----------
    num_components : int
        The maximum number of mixture components.
    covariance_type : MixtureCovarianceType
        The covariance structure of the components.
    weight_prior_type : BayesianGMMWeightPriorType
        The prior on the component weights.
    weight_concentration_prior : float | None
        The concentration of the weight prior; smaller values favor fewer
        active components. None uses ``1 / num_components``.
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

    num_components: int = 10
    covariance_type: MixtureCovarianceType = MixtureCovarianceType.FULL
    weight_prior_type: BayesianGMMWeightPriorType = (
        BayesianGMMWeightPriorType.DIRICHLET_PROCESS
    )
    weight_concentration_prior: float | None = None
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
        if (
            self.weight_concentration_prior is not None
            and self.weight_concentration_prior <= 0
        ):
            raise ValueError("weight_concentration_prior must be positive")
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
        estimator = BayesianGaussianMixture(
            n_components=self.num_components,
            covariance_type=self.covariance_type.value,
            weight_concentration_prior_type=self.weight_prior_type.value,
            weight_concentration_prior=self.weight_concentration_prior,
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
