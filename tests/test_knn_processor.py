from __future__ import annotations

import math
import unittest

import numpy as np

from density_estimation import DensityEstimationMethod, KNNMetric, KNNParameters
from tests.helpers import standard_normal


class TestKNNMetric(unittest.TestCase):
    def test_unit_ball_volumes(self) -> None:
        self.assertAlmostEqual(
            math.exp(KNNMetric.EUCLIDEAN.get_log_unit_ball_volume(2)), math.pi
        )
        self.assertAlmostEqual(
            math.exp(KNNMetric.EUCLIDEAN.get_log_unit_ball_volume(3)),
            4.0 / 3.0 * math.pi,
        )
        self.assertAlmostEqual(
            math.exp(KNNMetric.MANHATTAN.get_log_unit_ball_volume(2)), 2.0
        )
        self.assertAlmostEqual(
            math.exp(KNNMetric.CHEBYSHEV.get_log_unit_ball_volume(3)), 8.0
        )
        for metric in KNNMetric:
            self.assertAlmostEqual(math.exp(metric.get_log_unit_ball_volume(1)), 2.0)
        with self.assertRaises(ValueError):
            _ = KNNMetric.EUCLIDEAN.get_log_unit_ball_volume(0)


class TestKNNProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = KNNParameters(num_neighbors=0)

    def test_known_density_in_1d(self) -> None:
        processor = KNNParameters(num_neighbors=2).build_processor()
        processor.fit(np.array([[0.0], [1.0], [3.0], [10.0]]))
        scores = processor.score_samples(np.array([[0.5]]))
        # The 2nd nearest neighbor of 0.5 is at distance 0.5: p = 2 / (4 * 2 * 0.5).
        self.assertAlmostEqual(scores[0], math.log(0.5))
        self.assertEqual(processor.method, DensityEstimationMethod.KNN)

    def test_fit_score_samples_excludes_self(self) -> None:
        x = np.array([[0.0], [1.0], [3.0], [10.0]])
        processor = KNNParameters(num_neighbors=1).build_processor()
        scores = processor.fit_score_samples(x)
        # Leave-one-out: the nearest other sample of 0.0 is 1.0, so p = 1 / (3 * 2 * 1).
        self.assertAlmostEqual(scores[0], math.log(1.0 / 6.0))
        self.assertTrue(np.all(np.isfinite(scores.log_densities)))

    def test_outlier_has_lowest_density(self) -> None:
        x = np.vstack([standard_normal(100), [[10.0, 10.0]]])
        scores = KNNParameters(num_neighbors=5).build_processor().fit_score_samples(x)
        self.assertEqual(scores.get_lowest_indices(1), [100])

    def test_duplicate_query_gives_infinite_density(self) -> None:
        processor = KNNParameters(num_neighbors=1).build_processor()
        processor.fit(np.array([[0.0], [1.0]]))
        self.assertEqual(processor.score_samples(np.array([[0.0]]))[0], math.inf)

    def test_sample_count_requirements(self) -> None:
        processor = KNNParameters(num_neighbors=3).build_processor()
        with self.assertRaises(ValueError):
            processor.fit(np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            _ = processor.fit_score_samples(np.arange(3.0).reshape(3, 1))
        processor.fit(np.arange(3.0).reshape(3, 1))
        self.assertTrue(processor.is_fitted)

    def test_metrics_run(self) -> None:
        x = standard_normal(50)
        for metric in KNNMetric:
            scores = KNNParameters(metric=metric).build_processor().fit_score_samples(x)
            self.assertEqual(len(scores), 50)


if __name__ == "__main__":
    _ = unittest.main()
