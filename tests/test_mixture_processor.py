from __future__ import annotations

import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture

from density_estimation import (
    BayesianGMMParameters,
    BayesianGMMWeightPriorType,
    DensityEstimationMethod,
    GMMParameters,
    MixtureCovarianceType,
)
from tests.helpers import two_blobs


class TestGMMProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = GMMParameters(num_components=0)
        with self.assertRaises(ValueError):
            _ = GMMParameters(covariance_regularization=-1.0)
        with self.assertRaises(ValueError):
            _ = GMMParameters(num_initializations=0)

    def test_fit_and_components(self) -> None:
        x = two_blobs()
        processor = GMMParameters(num_components=2).build_processor()
        processor.fit(x)
        components = processor.components
        self.assertEqual(processor.method, DensityEstimationMethod.GMM)
        self.assertEqual(components.num_components, 2)
        self.assertEqual(components.num_features, 2)
        self.assertTrue(components.is_converged)
        self.assertAlmostEqual(float(np.sum(components.weights)), 1.0)
        np.testing.assert_allclose(
            np.sort(components.means[:, 0]), [0.0, 8.0], atol=0.5
        )

    def test_information_criteria_match_scikit_learn(self) -> None:
        x = two_blobs()
        for covariance_type in MixtureCovarianceType:
            processor = GMMParameters(
                num_components=3, covariance_type=covariance_type
            ).build_processor()
            processor.fit(x)
            criteria = processor.get_information_criteria(x)
            reference = GaussianMixture(
                n_components=3, covariance_type=covariance_type.value, random_state=0
            ).fit(x)
            self.assertAlmostEqual(criteria.bic, reference.bic(x), places=6)
            self.assertAlmostEqual(criteria.aic, reference.aic(x), places=6)

    def test_bic_prefers_true_component_count(self) -> None:
        x = two_blobs()
        bic_values: list[float] = []
        for num_components in (1, 2, 4):
            processor = GMMParameters(num_components=num_components).build_processor()
            processor.fit(x)
            bic_values.append(processor.get_information_criteria(x).bic)
        self.assertEqual(int(np.argmin(bic_values)), 1)

    def test_full_covariances_for_each_type(self) -> None:
        x = two_blobs()
        for covariance_type in MixtureCovarianceType:
            processor = GMMParameters(covariance_type=covariance_type).build_processor()
            processor.fit(x)
            full = processor.components.full_covariances
            self.assertEqual(full.shape, (2, 2, 2))
            np.testing.assert_allclose(full, np.transpose(full, (0, 2, 1)))

    def test_component_probabilities(self) -> None:
        x = two_blobs()
        processor = GMMParameters().build_processor()
        processor.fit(x)
        probabilities = processor.predict_component_probabilities(x)
        self.assertEqual(probabilities.shape, (x.shape[0], 2))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)

    def test_sample_is_shuffled_and_reproducible(self) -> None:
        processor = GMMParameters().build_processor()
        processor.fit(two_blobs())
        samples = processor.sample(200)
        again = processor.sample(200)
        np.testing.assert_array_equal(samples.samples, again.samples)
        first_half_mean = float(np.mean(samples.samples[:100, 0]))
        self.assertTrue(1.0 < first_half_mean < 7.0)

    def test_requires_enough_samples(self) -> None:
        processor = GMMParameters(num_components=5).build_processor()
        with self.assertRaises(ValueError):
            processor.fit(two_blobs()[:3])

    def test_requires_fit(self) -> None:
        processor = GMMParameters().build_processor()
        with self.assertRaises(NotFittedError):
            _ = processor.components
        with self.assertRaises(NotFittedError):
            _ = processor.predict_component_probabilities(two_blobs())


class TestBayesianGMMProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = BayesianGMMParameters(num_components=0)
        with self.assertRaises(ValueError):
            _ = BayesianGMMParameters(weight_concentration_prior=0.0)

    def test_prunes_unneeded_components(self) -> None:
        x = two_blobs()
        processor = BayesianGMMParameters(
            num_components=6,
            weight_prior_type=BayesianGMMWeightPriorType.DIRICHLET_PROCESS,
            weight_concentration_prior=0.01,
            max_iterations=500,
        ).build_processor()
        scores = processor.fit_score_samples(x)
        components = processor.components
        self.assertEqual(processor.method, DensityEstimationMethod.BAYESIAN_GMM)
        self.assertEqual(len(scores), x.shape[0])
        self.assertEqual(components.num_components, 6)
        self.assertEqual(len(components.get_effective_component_indices(0.05)), 2)

    def test_sample_and_information_criteria(self) -> None:
        x = two_blobs()
        processor = BayesianGMMParameters(
            num_components=3, max_iterations=500
        ).build_processor()
        processor.fit(x)
        self.assertEqual(processor.sample(10).num_samples, 10)
        criteria = processor.get_information_criteria(x)
        self.assertEqual(criteria.num_samples, x.shape[0])


if __name__ == "__main__":
    _ = unittest.main()
