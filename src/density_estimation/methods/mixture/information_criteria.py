from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InformationCriteria:
    """
    Model selection criteria of a fitted density on a dataset.

    Lower values of AIC and BIC indicate a better trade-off between fit and
    model complexity; compare them only across models scored on the same data.

    Attributes
    ----------
    log_likelihood : float
        The total log-likelihood of the data.
    num_samples : int
        The number of samples the log-likelihood was computed on.
    num_free_parameters : int
        The number of free parameters of the model.
    """

    log_likelihood: float
    num_samples: int
    num_free_parameters: int

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be at least 1, got {self.num_samples}")
        if self.num_free_parameters < 0:
            raise ValueError(
                f"num_free_parameters must be non-negative, got {self.num_free_parameters}"
            )

    @property
    def aic(self) -> float:
        """
        Get the Akaike information criterion.

        Returns
        -------
        float
            ``2 * k - 2 * log_likelihood``.
        """
        return 2.0 * self.num_free_parameters - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        """
        Get the Bayesian information criterion.

        Returns
        -------
        float
            ``k * ln(n) - 2 * log_likelihood``.
        """
        return (
            self.num_free_parameters * math.log(self.num_samples)
            - 2.0 * self.log_likelihood
        )
