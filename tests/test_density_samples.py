from __future__ import annotations

import unittest

import numpy as np

from density_estimation import DensitySamples


class TestDensitySamples(unittest.TestCase):
    def test_statistics(self) -> None:
        samples = DensitySamples([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        self.assertEqual(samples.num_samples, 3)
        self.assertEqual(samples.num_features, 2)
        self.assertEqual(len(samples), 3)
        np.testing.assert_allclose(samples.mean, [2.0, 3.0])
        np.testing.assert_allclose(samples.covariance, [[4.0, 4.0], [4.0, 4.0]])
        np.testing.assert_allclose(samples.lower_bounds, [0.0, 1.0])
        np.testing.assert_allclose(samples.upper_bounds, [4.0, 5.0])

    def test_single_feature_covariance_is_2d(self) -> None:
        samples = DensitySamples([[0.0], [2.0]])
        self.assertEqual(samples.covariance.shape, (1, 1))

    def test_covariance_requires_two_samples(self) -> None:
        with self.assertRaises(ValueError):
            _ = DensitySamples([[0.0, 1.0]]).covariance

    def test_rejects_invalid_samples(self) -> None:
        with self.assertRaises(ValueError):
            _ = DensitySamples([0.0, 1.0])
        with self.assertRaises(ValueError):
            _ = DensitySamples([[np.inf]])

    def test_is_read_only(self) -> None:
        samples = DensitySamples([[0.0]])
        with self.assertRaises(ValueError):
            samples.samples[0, 0] = 1.0


if __name__ == "__main__":
    _ = unittest.main()
