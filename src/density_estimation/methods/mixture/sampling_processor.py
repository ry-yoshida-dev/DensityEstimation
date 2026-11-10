from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from ...method import DensityEstimationMethod
from ...types import FloatArray, NumericArray
from ..sampling_processor import SamplingDensityProcessor
from .components import MixtureComponents
from .covariance_type import MixtureCovarianceType
from .information_criteria import InformationCriteria


@dataclass
class MixtureProcessor(SamplingDensityProcessor):
    """
    Wraps scikit-learn's GaussianMixture and BayesianGaussianMixture.

    Attributes
    ----------
    estimator : GaussianMixture | BayesianGaussianMixture
        The underlying estimator.
    covariance_type : MixtureCovarianceType
        The covariance structure the estimator is configured with.
    random_state : int | None
        The seed for shuffling drawn samples, which scikit-learn returns
        grouped by component.
    """

    estimator: GaussianMixture | BayesianGaussianMixture
    covariance_type: MixtureCovarianceType
    random_state: int | None

    @property
    @override
    def method(self) -> DensityEstimationMethod:
        match self.estimator:
            case BayesianGaussianMixture():
                return DensityEstimationMethod.BAYESIAN_GMM
            case GaussianMixture():
                return DensityEstimationMethod.GMM

    @property
    def components(self) -> MixtureComponents:
        """
        Get the fitted mixture parameters.

        Returns
        -------
        MixtureComponents
            The fitted weights, means, covariances, and convergence state.

        Raises
        ------
        NotFittedError
            If the processor has not been fitted yet.
        """
        _ = self.num_features
        return MixtureComponents(
            weights=self.estimator.weights_,
            means=self.estimator.means_,
            covariances=self.estimator.covariances_,
            covariance_type=self.covariance_type,
            is_converged=bool(self.estimator.converged_),
            num_iterations=int(self.estimator.n_iter_),
            lower_bound=float(self.estimator.lower_bound_),
        )

    def get_information_criteria(self, X: NumericArray) -> InformationCriteria:
        """
        Compute AIC and BIC of the fitted mixture on a dataset.

        The criteria assume a maximum likelihood fit, which only a GMM
        provides. For a Bayesian GMM they are rough: the parameter count
        includes components the prior has driven to negligible weight, so
        compare Bayesian mixtures by ``components.lower_bound`` instead.

        Parameters
        ----------
        X : NumericArray
            The samples with shape (n_samples, n_features), usually the
            training samples.

        Returns
        -------
        InformationCriteria
            The log-likelihood, sample count, and parameter count of the
            mixture on X.

        Raises
        ------
        NotFittedError
            If the processor has not been fitted yet.
        ValueError
            If X is not a non-empty finite 2-D array with the fitted number of
            features.
        """
        scores = self.score_samples(X)
        return InformationCriteria(
            log_likelihood=scores.total_log_likelihood,
            num_samples=len(scores),
            num_free_parameters=self.components.num_free_parameters,
        )

    def predict_component_probabilities(self, X: NumericArray) -> FloatArray:
        """
        Compute the posterior probability of each component per sample.

        Parameters
        ----------
        X : NumericArray
            The samples with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The responsibilities with shape (n_samples, n_components); each
            row sums to one.

        Raises
        ------
        NotFittedError
            If the processor has not been fitted yet.
        ValueError
            If X is not a non-empty finite 2-D array with the fitted number of
            features.
        """
        samples = self._as_fitted_sample_matrix(X)
        return np.asarray(self.estimator.predict_proba(samples), dtype=np.float64)

    @override
    def _fit(self, samples: FloatArray) -> None:
        num_components = self.estimator.n_components
        if samples.shape[0] < num_components:
            raise ValueError(
                f"X must have at least num_components={num_components} samples, got {samples.shape[0]}"
            )
        _ = self.estimator.fit(samples)

    @override
    def _score_samples(self, samples: FloatArray) -> FloatArray:
        return np.asarray(self.estimator.score_samples(samples), dtype=np.float64)

    @override
    def _sample(self, num_samples: int) -> FloatArray:
        drawn, _ = self.estimator.sample(num_samples)
        permutation = np.random.default_rng(self.random_state).permutation(num_samples)
        return np.asarray(drawn, dtype=np.float64)[permutation]
