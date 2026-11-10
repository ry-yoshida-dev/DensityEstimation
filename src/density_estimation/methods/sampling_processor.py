from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..results import DensitySamples
from ..types import FloatArray
from .processor import DensityProcessor


@dataclass
class SamplingDensityProcessor(DensityProcessor, ABC):
    """
    Density processor that can also draw samples from the fitted density.

    Sampling is seeded by the random state of the method's parameters, so
    repeated calls with the same sample count return the same samples when
    that random state is an integer.
    """

    def sample(self, num_samples: int) -> DensitySamples:
        """
        Draw samples from the fitted density.

        Parameters
        ----------
        num_samples : int
            The number of samples to draw, at least 1.

        Returns
        -------
        DensitySamples
            The drawn samples.

        Raises
        ------
        ValueError
            If num_samples is below 1.
        NotFittedError
            If the processor has not been fitted yet.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be at least 1, got {num_samples}")
        _ = self.num_features
        return DensitySamples(self._sample(num_samples))

    @abstractmethod
    def _sample(self, num_samples: int) -> FloatArray: ...
