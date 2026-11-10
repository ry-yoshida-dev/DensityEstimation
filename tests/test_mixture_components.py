from __future__ import annotations

import math
import unittest

import numpy as np

from density_estimation import (
    InformationCriteria,
    MixtureComponents,
    MixtureCovarianceType,
)


def build(
    covariance_type: MixtureCovarianceType, covariances: list[float] | list[list[float]]
) -> MixtureComponents:
    return MixtureComponents(
        weights=[0.7, 0.3],
        means=[[0.0, 0.0], [1.0, 1.0]],
        covariances=covariances,
        covariance_type=covariance_type,
        is_converged=True,
        num_iterations=3,
        lower_bound=-1.0,
    )


class TestMixtureComponents(unittest.TestCase):
    def test_spherical_and_diag_expand_to_full(self) -> None:
        spherical = build(MixtureCovarianceType.SPHERICAL, [2.0, 3.0])
        np.testing.assert_allclose(
            spherical.full_covariances[1, :, :], [[3.0, 0.0], [0.0, 3.0]]
        )
        diag = build(MixtureCovarianceType.DIAG, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            diag.full_covariances[0, :, :], [[1.0, 0.0], [0.0, 2.0]]
        )

    def test_tied_expands_to_every_component(self) -> None:
        tied = build(MixtureCovarianceType.TIED, [[1.0, 0.5], [0.5, 1.0]])
        full = tied.full_covariances
        self.assertEqual(full.shape, (2, 2, 2))
        np.testing.assert_allclose(full[0, :, :], full[1, :, :])

    def test_num_free_parameters(self) -> None:
        self.assertEqual(
            build(MixtureCovarianceType.SPHERICAL, [2.0, 3.0]).num_free_parameters,
            2 + 4 + 1,
        )
        self.assertEqual(
            build(
                MixtureCovarianceType.TIED, [[1.0, 0.0], [0.0, 1.0]]
            ).num_free_parameters,
            3 + 4 + 1,
        )

    def test_rejects_mismatched_covariance_shape(self) -> None:
        with self.assertRaises(ValueError):
            _ = build(MixtureCovarianceType.FULL, [2.0, 3.0])

    def test_effective_components(self) -> None:
        components = build(MixtureCovarianceType.SPHERICAL, [2.0, 3.0])
        self.assertEqual(components.get_effective_component_indices(0.0), [0, 1])
        self.assertEqual(components.get_effective_component_indices(0.5), [0])
        with self.assertRaises(ValueError):
            _ = components.get_effective_component_indices(1.5)


class TestInformationCriteria(unittest.TestCase):
    def test_values(self) -> None:
        criteria = InformationCriteria(
            log_likelihood=-100.0, num_samples=50, num_free_parameters=5
        )
        self.assertAlmostEqual(criteria.aic, 210.0)
        self.assertAlmostEqual(criteria.bic, 5 * math.log(50) + 200.0)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = InformationCriteria(
                log_likelihood=0.0, num_samples=0, num_free_parameters=1
            )
        with self.assertRaises(ValueError):
            _ = InformationCriteria(
                log_likelihood=0.0, num_samples=1, num_free_parameters=-1
            )


if __name__ == "__main__":
    _ = unittest.main()
