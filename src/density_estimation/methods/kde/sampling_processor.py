from __future__ import annotations

from dataclasses import dataclass
from typing import override

from sklearn.neighbors import KernelDensity

from ...method import DensityEstimationMethod
from ...types import FloatArray
from ..sampling_processor import SamplingDensityProcessor
from .bandwidth_rule import KDEBandwidthRule
from .kernel import KDEKernel


@dataclass
class KDEProcessor(SamplingDensityProcessor):
    """
    Wraps scikit-learn's KernelDensity.

    Training scores from fit_score_samples include each sample's own kernel
    contribution, so they are optimistic compared with scores of unseen data.

    Attributes
    ----------
    estimator : KernelDensity
        The underlying estimator. Its bandwidth is set on every fit.
    bandwidth_setting : float | KDEBandwidthRule
        The fixed bandwidth, or the rule that derives it on every fit.
    kernel : KDEKernel
        The kernel the estimator is configured with.
    random_state : int | None
        The seed for sampling.
    """

    estimator: KernelDensity
    bandwidth_setting: float | KDEBandwidthRule
    kernel: KDEKernel
    random_state: int | None

    @property
    @override
    def method(self) -> DensityEstimationMethod:
        return DensityEstimationMethod.KDE

    @property
    def bandwidth(self) -> float:
        """
        Get the bandwidth used by the fitted estimator.

        Resolves a bandwidth rule to its value for the training data.

        Returns
        -------
        float
            The fitted bandwidth.

        Raises
        ------
        NotFittedError
            If the processor has not been fitted yet.
        """
        _ = self.num_features
        return float(self.estimator.bandwidth_)

    @override
    def _fit(self, samples: FloatArray) -> None:
        if isinstance(self.bandwidth_setting, KDEBandwidthRule):
            self.estimator.bandwidth = self.bandwidth_setting.get_bandwidth(samples)
        else:
            self.estimator.bandwidth = self.bandwidth_setting
        _ = self.estimator.fit(samples)

    @override
    def _score_samples(self, samples: FloatArray) -> FloatArray:
        return self.estimator.score_samples(samples)

    @override
    def _sample(self, num_samples: int) -> FloatArray:
        if not self.kernel.is_sampling_supported:
            raise NotImplementedError(
                f"sampling is not supported for the {self.kernel.value} kernel; use {KDEKernel.GAUSSIAN.value} or {KDEKernel.TOPHAT.value}"
            )
        return self.estimator.sample(num_samples, random_state=self.random_state)
