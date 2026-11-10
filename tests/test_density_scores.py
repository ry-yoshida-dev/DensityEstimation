from __future__ import annotations

import math
import unittest

import numpy as np

from density_estimation import DensityScores

SCORES = DensityScores([-1.0, -5.0, -2.0, -10.0, -0.5])


class TestDensityScores(unittest.TestCase):
    def test_rejects_empty_and_nan(self) -> None:
        with self.assertRaises(ValueError):
            _ = DensityScores([])
        with self.assertRaises(ValueError):
            _ = DensityScores([0.0, float("nan")])

    def test_is_read_only_copy(self) -> None:
        source = np.array([-1.0, -2.0])
        scores = DensityScores(source)
        source[0] = 0.0
        self.assertEqual(scores[0], -1.0)
        with self.assertRaises(ValueError):
            scores.log_densities[0] = 1.0

    def test_summary_statistics(self) -> None:
        self.assertAlmostEqual(SCORES.total_log_likelihood, -18.5)
        self.assertAlmostEqual(SCORES.mean_log_likelihood, -3.7)
        self.assertEqual(SCORES.min_log_density, -10.0)
        self.assertEqual(SCORES.max_log_density, -0.5)
        np.testing.assert_allclose(SCORES.densities, np.exp(SCORES.log_densities))

    def test_probability_weights_sum_to_one_for_tiny_densities(self) -> None:
        scores = DensityScores([-1000.0, -1000.0 + math.log(3.0)])
        np.testing.assert_allclose(scores.probability_weights, [0.25, 0.75])

    def test_probability_weights_reject_all_zero_density(self) -> None:
        with self.assertRaises(ValueError):
            _ = DensityScores([-math.inf, -math.inf]).probability_weights

    def test_highest_and_lowest_indices(self) -> None:
        self.assertEqual(SCORES.get_highest_indices(2), [4, 0])
        self.assertEqual(SCORES.get_lowest_indices(2), [3, 1])
        with self.assertRaises(ValueError):
            _ = SCORES.get_highest_indices(0)
        with self.assertRaises(ValueError):
            _ = SCORES.get_lowest_indices(6)

    def test_outliers(self) -> None:
        self.assertEqual(SCORES.get_outlier_indices(0.4), [1, 3])
        mask = SCORES.get_outlier_mask(0.1)
        self.assertEqual(mask.tolist(), [False, False, False, True, False])
        with self.assertRaises(ValueError):
            _ = SCORES.get_outlier_mask(0.0)
        with self.assertRaises(ValueError):
            _ = SCORES.get_outlier_mask(0.6)

    def test_thresholds(self) -> None:
        self.assertEqual(SCORES.get_quantile_threshold(0.5), -2.0)
        self.assertEqual(
            SCORES.get_below_threshold_mask(-2.0).tolist(),
            [False, True, False, True, False],
        )
        with self.assertRaises(ValueError):
            _ = SCORES.get_quantile_threshold(1.5)

    def test_sequence_protocol(self) -> None:
        self.assertEqual(len(SCORES), 5)
        self.assertEqual(list(SCORES)[0], -1.0)
        self.assertEqual(np.asarray(SCORES).shape, (5,))
        with self.assertRaises(IndexError):
            _ = SCORES[5]

    def test_equality_and_hash(self) -> None:
        other = DensityScores([-1.0, -5.0, -2.0, -10.0, -0.5])
        self.assertEqual(SCORES, other)
        self.assertEqual(hash(SCORES), hash(other))
        self.assertNotEqual(SCORES, DensityScores([0.0]))

    def test_signed_zeros_hash_equally(self) -> None:
        positive = DensityScores([0.0, 1.0])
        negative = DensityScores([-0.0, 1.0])
        self.assertEqual(positive, negative)
        self.assertEqual(hash(positive), hash(negative))


if __name__ == "__main__":
    _ = unittest.main()
