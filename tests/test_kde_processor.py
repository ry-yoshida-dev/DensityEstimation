from __future__ import annotations

import math
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from density_estimation import (
    DensityEstimationMethod,
    GridAxes,
    KDEBandwidthRule,
    KDEKernel,
    KDEParameters,
)
from tests.helpers import standard_normal


class TestKDEProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = KDEParameters(bandwidth=0.0)
        with self.assertRaises(ValueError):
            _ = KDEParameters(bandwidth=math.nan)
        with self.assertRaises(ValueError):
            _ = KDEParameters(leaf_size=0)
        with self.assertRaises(ValueError):
            _ = KDEParameters(absolute_tolerance=-1.0)

    def test_fit_score_samples(self) -> None:
        x = standard_normal()
        processor = KDEParameters(bandwidth=0.5).build_processor()
        scores = processor.fit_score_samples(x)
        self.assertEqual(processor.method, DensityEstimationMethod.KDE)
        self.assertEqual(len(scores), x.shape[0])
        self.assertTrue(processor.is_fitted)
        self.assertEqual(processor.num_features, 2)

    def test_center_is_denser_than_tail(self) -> None:
        processor = KDEParameters().build_processor()
        processor.fit(standard_normal(500))
        scores = processor.score_samples(np.array([[0.0, 0.0], [4.0, 4.0]]))
        self.assertEqual(scores.get_highest_indices(1), [0])

    def test_bandwidth_rule_is_resolved(self) -> None:
        x = standard_normal()
        processor = KDEParameters(bandwidth=KDEBandwidthRule.SCOTT).build_processor()
        processor.fit(x)
        scale = float(np.mean(np.std(x, axis=0, ddof=1)))
        self.assertAlmostEqual(
            processor.bandwidth, math.pow(len(x), -1.0 / 6.0) * scale
        )

    def test_bandwidth_rule_follows_data_scale(self) -> None:
        x = standard_normal()
        for rule in KDEBandwidthRule:
            unit = KDEParameters(bandwidth=rule).build_processor()
            unit.fit(x)
            scaled = KDEParameters(bandwidth=rule).build_processor()
            scaled.fit(x * 100.0)
            self.assertAlmostEqual(scaled.bandwidth, unit.bandwidth * 100.0)

    def test_bandwidth_rule_rejects_constant_samples(self) -> None:
        processor = KDEParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(np.ones((10, 2)))

    def test_grid_integrates_to_one(self) -> None:
        processor = KDEParameters(bandwidth=0.5).build_processor()
        processor.fit(standard_normal())
        grid = processor.evaluate_grid(
            GridAxes.from_bounds([-7.0, -7.0], [7.0, 7.0], num_points=141)
        )
        self.assertAlmostEqual(grid.probability_mass, 1.0, places=3)

    def test_sample_is_reproducible(self) -> None:
        processor = KDEParameters(random_state=3).build_processor()
        processor.fit(standard_normal())
        first = processor.sample(50)
        second = processor.sample(50)
        self.assertEqual(first.samples.shape, (50, 2))
        np.testing.assert_array_equal(first.samples, second.samples)

    def test_sample_rejects_unsupported_kernel(self) -> None:
        self.assertFalse(KDEKernel.EPANECHNIKOV.is_sampling_supported)
        processor = KDEParameters(kernel=KDEKernel.EPANECHNIKOV).build_processor()
        processor.fit(standard_normal())
        with self.assertRaises(NotImplementedError):
            _ = processor.sample(1)

    def test_compact_kernel_gives_zero_density_far_away(self) -> None:
        processor = KDEParameters(
            bandwidth=0.5, kernel=KDEKernel.TOPHAT
        ).build_processor()
        processor.fit(standard_normal())
        scores = processor.score_samples(np.array([[100.0, 100.0]]))
        self.assertEqual(scores[0], -math.inf)

    def test_requires_fit(self) -> None:
        processor = KDEParameters().build_processor()
        self.assertFalse(processor.is_fitted)
        with self.assertRaises(NotFittedError):
            _ = processor.score_samples(standard_normal())
        with self.assertRaises(NotFittedError):
            _ = processor.sample(1)
        with self.assertRaises(NotFittedError):
            _ = processor.bandwidth

    def test_rejects_invalid_input(self) -> None:
        processor = KDEParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(np.zeros(5))
        with self.assertRaises(ValueError):
            processor.fit(np.array([[np.nan, 0.0]]))
        processor.fit(standard_normal())
        with self.assertRaises(ValueError):
            _ = processor.score_samples(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            _ = processor.evaluate_grid(GridAxes([[0.0, 1.0]]))
        with self.assertRaises(ValueError):
            _ = processor.sample(0)


if __name__ == "__main__":
    _ = unittest.main()
