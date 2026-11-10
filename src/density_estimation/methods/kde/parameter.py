import math
from dataclasses import dataclass
from typing import override

from sklearn.neighbors import KernelDensity

from ..parameter import DensityParameters
from .algorithm import KDEAlgorithm
from .bandwidth_rule import KDEBandwidthRule
from .kernel import KDEKernel
from .sampling_processor import KDEProcessor


@dataclass(frozen=True)
class KDEParameters(DensityParameters):
    """
    Parameters for scikit-learn's KernelDensity.

    Attributes
    ----------
    bandwidth : float | KDEBandwidthRule
        The kernel bandwidth, or the rule that derives it from the scale and
        size of the training data.
    kernel : KDEKernel
        The kernel. Only the Gaussian and top-hat kernels support sampling.
    algorithm : KDEAlgorithm
        The tree used to accelerate evaluation.
    absolute_tolerance : float
        The absolute tolerance of the tree approximation; 0 is exact.
    relative_tolerance : float
        The relative tolerance of the tree approximation; 0 is exact.
    leaf_size : int
        The leaf size of the tree.
    random_state : int | None
        The seed for sampling.
    """

    bandwidth: float | KDEBandwidthRule = KDEBandwidthRule.SCOTT
    kernel: KDEKernel = KDEKernel.GAUSSIAN
    algorithm: KDEAlgorithm = KDEAlgorithm.AUTO
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    leaf_size: int = 40
    random_state: int | None = 0

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if not isinstance(self.bandwidth, KDEBandwidthRule) and not (
            0 < self.bandwidth < math.inf
        ):
            raise ValueError(
                f"bandwidth must be positive and finite, got {self.bandwidth}"
            )
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be non-negative")
        if self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative")
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be greater than 0")

    @override
    def build_processor(self) -> KDEProcessor:
        estimator = KernelDensity(
            algorithm=self.algorithm.value,
            kernel=self.kernel.value,
            atol=self.absolute_tolerance,
            rtol=self.relative_tolerance,
            leaf_size=self.leaf_size,
        )
        return KDEProcessor(
            estimator=estimator,
            bandwidth_setting=self.bandwidth,
            kernel=self.kernel,
            random_state=self.random_state,
        )
