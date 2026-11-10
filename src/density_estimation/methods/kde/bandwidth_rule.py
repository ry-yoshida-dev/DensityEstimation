import math
from enum import Enum

import numpy as np

from ...types import FloatArray


class KDEBandwidthRule(Enum):
    """
    Rule of thumb that derives the KDE bandwidth from the training data.

    Each rule multiplies a sample-size factor by the data scale, the mean of
    the per-feature sample standard deviations. The kernel is isotropic, so
    standardize features whose scales differ widely. Both rules assume
    roughly Gaussian data and oversmooth multimodal data.

    Attributes
    ----------
    SCOTT
        Scott's rule, with factor ``n ** (-1 / (d + 4))``.
    SILVERMAN
        Silverman's rule, with factor ``(n * (d + 2) / 4) ** (-1 / (d + 4))``.
    """

    SCOTT = "scott"
    SILVERMAN = "silverman"

    def get_bandwidth(self, samples: FloatArray) -> float:
        """
        Compute the bandwidth for the given training samples.

        Parameters
        ----------
        samples : FloatArray
            The training samples with shape (n_samples, n_features).

        Returns
        -------
        float
            The positive bandwidth.

        Raises
        ------
        ValueError
            If there are fewer than two samples or every feature is constant.
        """
        num_samples: int = np.size(samples, 0)
        num_features: int = np.size(samples, 1)
        if num_samples < 2:
            raise ValueError(
                f"a bandwidth rule requires at least 2 samples, got {num_samples}"
            )
        scale = float(np.mean(np.std(samples, axis=0, ddof=1)))
        if scale <= 0.0:
            raise ValueError(
                "a bandwidth rule requires at least one non-constant feature"
            )
        exponent = -1.0 / (num_features + 4)
        match self:
            case KDEBandwidthRule.SCOTT:
                factor = math.pow(num_samples, exponent)
            case KDEBandwidthRule.SILVERMAN:
                factor = math.pow(num_samples * (num_features + 2) / 4.0, exponent)
        return factor * scale
