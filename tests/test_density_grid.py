from __future__ import annotations

import math
import unittest

import numpy as np

from density_estimation import DensityGrid, GridAxes


def standard_normal_grid() -> DensityGrid:
    axes = GridAxes.from_bounds([-6.0, -6.0], [6.0, 6.0], num_points=121)
    points = axes.points
    log_densities = -0.5 * np.sum(points**2, axis=1) - math.log(2.0 * math.pi)
    return DensityGrid(axes, log_densities)


class TestDensityGrid(unittest.TestCase):
    def test_rejects_wrong_size(self) -> None:
        axes = GridAxes([[0.0, 1.0]])
        with self.assertRaises(ValueError):
            _ = DensityGrid(axes, [0.0, 0.0, 0.0])

    def test_probability_mass_and_mode(self) -> None:
        grid = standard_normal_grid()
        self.assertEqual(grid.log_densities.shape, (121, 121))
        self.assertAlmostEqual(grid.probability_mass, 1.0, places=4)
        np.testing.assert_allclose(grid.mode, [0.0, 0.0], atol=1e-12)

    def test_highest_density_region(self) -> None:
        grid = standard_normal_grid()
        mask = grid.get_highest_density_region_mask(0.5)
        captured = float(np.sum(grid.densities[mask]) * grid.axes.cell_volume)
        self.assertAlmostEqual(captured, 0.5, places=2)
        self.assertTrue(np.all(grid.get_highest_density_region_mask(1.0)))
        with self.assertRaises(ValueError):
            _ = grid.get_highest_density_threshold(0.0)

    def test_marginalize_gives_normalized_marginal(self) -> None:
        marginal = standard_normal_grid().marginalize(1)
        self.assertEqual(marginal.axes.num_dimensions, 1)
        self.assertAlmostEqual(marginal.probability_mass, 1.0, places=4)
        expected = -0.5 * marginal.axes.coordinates[0] ** 2 - 0.5 * math.log(
            2.0 * math.pi
        )
        np.testing.assert_allclose(marginal.log_densities, expected, atol=1e-6)

    def test_marginalize_keeps_zero_density(self) -> None:
        axes = GridAxes([[0.0, 1.0], [0.0, 1.0]])
        grid = DensityGrid(axes, [[-math.inf, -math.inf], [0.0, 0.0]])
        marginal = grid.marginalize(1)
        self.assertEqual(marginal.log_densities[0], -math.inf)
        self.assertAlmostEqual(marginal.log_densities.item(1), math.log(2.0))

    def test_marginalize_rejects_1d(self) -> None:
        grid = DensityGrid(GridAxes([[0.0, 1.0]]), [0.0, 0.0])
        with self.assertRaises(ValueError):
            _ = grid.marginalize(0)


if __name__ == "__main__":
    _ = unittest.main()
