from __future__ import annotations

import unittest

import numpy as np

from density_estimation import GridAxes


class TestGridAxes(unittest.TestCase):
    def test_from_bounds(self) -> None:
        axes = GridAxes.from_bounds([0.0, -1.0], [1.0, 1.0], num_points=5)
        self.assertEqual(axes.shape, (5, 5))
        self.assertEqual(axes.num_points, 25)
        self.assertEqual(axes.num_dimensions, 2)
        np.testing.assert_allclose(axes.spacings, [0.25, 0.5])
        self.assertAlmostEqual(axes.cell_volume, 0.125)

    def test_points_follow_c_order(self) -> None:
        axes = GridAxes([[0.0, 1.0], [10.0, 20.0, 30.0]])
        np.testing.assert_allclose(
            axes.points,
            [[0, 10], [0, 20], [0, 30], [1, 10], [1, 20], [1, 30]],
        )

    def test_from_samples_adds_padding(self) -> None:
        axes = GridAxes.from_samples([[0.0, 0.0], [10.0, 2.0]], num_points=3)
        np.testing.assert_allclose(axes.coordinates[0], [-1.0, 5.0, 11.0])
        np.testing.assert_allclose(axes.coordinates[1], [-0.2, 1.0, 2.2])

    def test_from_samples_rejects_constant_feature(self) -> None:
        with self.assertRaises(ValueError):
            _ = GridAxes.from_samples([[0.0, 1.0], [0.0, 2.0]], num_points=3)

    def test_rejects_invalid_axes(self) -> None:
        with self.assertRaises(ValueError):
            _ = GridAxes([])
        with self.assertRaises(ValueError):
            _ = GridAxes([[0.0]])
        with self.assertRaises(ValueError):
            _ = GridAxes([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            _ = GridAxes([[0.0, 1.0, 3.0]])
        with self.assertRaises(ValueError):
            _ = GridAxes.from_bounds([0.0], [1.0, 2.0], num_points=3)
        with self.assertRaises(ValueError):
            _ = GridAxes.from_bounds([0.0], [1.0], num_points=1)

    def test_without_axis(self) -> None:
        axes = GridAxes([[0.0, 1.0], [5.0, 6.0, 7.0]])
        self.assertEqual(axes.without_axis(0).shape, (3,))
        with self.assertRaises(IndexError):
            _ = axes.without_axis(2)
        with self.assertRaises(ValueError):
            _ = axes.without_axis(0).without_axis(0)


if __name__ == "__main__":
    _ = unittest.main()
