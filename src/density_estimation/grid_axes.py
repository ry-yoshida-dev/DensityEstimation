from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray
from .validation import as_sample_matrix


@dataclass(frozen=True, eq=False)
class GridAxes:
    """
    Uniformly spaced coordinates of a rectangular evaluation grid.

    Attributes
    ----------
    coordinates : tuple[FloatArray, ...]
        One read-only, strictly increasing, uniformly spaced 1-D array per
        dimension.
    """

    coordinates: tuple[FloatArray, ...] = field(init=False, repr=True)

    def __init__(self, coordinates: Sequence[ArrayLike]) -> None:
        if len(coordinates) == 0:
            raise ValueError("coordinates must contain at least one axis")
        owned_axes: list[FloatArray] = []
        for axis_index, axis_coordinates in enumerate(coordinates):
            owned = np.array(axis_coordinates, dtype=np.float64)
            self._validate_axis(axis_index, owned)
            owned.setflags(write=False)
            owned_axes.append(owned)
        object.__setattr__(self, "coordinates", tuple(owned_axes))

    @staticmethod
    def _validate_axis(axis_index: int, axis_coordinates: FloatArray) -> None:
        if axis_coordinates.ndim != 1:
            raise ValueError(
                f"axis {axis_index} must be 1-D, got {axis_coordinates.ndim}-D"
            )
        if axis_coordinates.size < 2:
            raise ValueError(
                f"axis {axis_index} must have at least 2 points, got {axis_coordinates.size}"
            )
        if not np.all(np.isfinite(axis_coordinates)):
            raise ValueError(f"axis {axis_index} must contain only finite values")
        steps = np.diff(axis_coordinates)
        if np.any(steps <= 0.0):
            raise ValueError(f"axis {axis_index} must be strictly increasing")
        first_step: float = steps.item(0)
        if not np.allclose(steps, first_step, rtol=1e-6, atol=0.0):
            raise ValueError(f"axis {axis_index} must be uniformly spaced")

    @classmethod
    def from_bounds(
        cls,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        num_points: int,
    ) -> GridAxes:
        """
        Build axes spanning the given bounds.

        Parameters
        ----------
        lower_bounds : Sequence[float]
            The lower bound of each dimension.
        upper_bounds : Sequence[float]
            The upper bound of each dimension.
        num_points : int
            The number of points per axis, at least 2.

        Returns
        -------
        GridAxes
            Axes whose first and last coordinates equal the bounds.

        Raises
        ------
        ValueError
            If the bounds differ in length or num_points is below 2. Bounds
            that are not strictly increasing are rejected by the constructor.
        """
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError(
                f"lower_bounds and upper_bounds must have the same length, got {len(lower_bounds)} and {len(upper_bounds)}"
            )
        if num_points < 2:
            raise ValueError(f"num_points must be at least 2, got {num_points}")
        return cls(
            [
                np.linspace(lower, upper, num_points)
                for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
            ]
        )

    @classmethod
    def from_samples(
        cls,
        samples: ArrayLike,
        num_points: int,
        padding_ratio: float = 0.1,
    ) -> GridAxes:
        """
        Build axes covering the range of the samples with a margin.

        Parameters
        ----------
        samples : ArrayLike
            The samples with shape (n_samples, n_features).
        num_points : int
            The number of points per axis, at least 2.
        padding_ratio : float
            The margin added on each side, as a fraction of each feature's
            range. Must be non-negative.

        Returns
        -------
        GridAxes
            Axes spanning ``[min - padding, max + padding]`` per feature.

        Raises
        ------
        ValueError
            If padding_ratio is negative, the samples are invalid, or any
            feature is constant.
        """
        if padding_ratio < 0.0:
            raise ValueError(f"padding_ratio must be non-negative, got {padding_ratio}")
        matrix = as_sample_matrix(samples, name="samples")
        lower = np.min(matrix, axis=0)
        upper = np.max(matrix, axis=0)
        span = upper - lower
        constant_features = np.flatnonzero(span <= 0.0)
        if constant_features.size > 0:
            raise ValueError(
                f"features {constant_features.tolist()} are constant, so their range is undefined"
            )
        padding = padding_ratio * span
        return cls.from_bounds(
            cast(list[float], (lower - padding).tolist()),
            cast(list[float], (upper + padding).tolist()),
            num_points,
        )

    @property
    def num_dimensions(self) -> int:
        """
        Get the number of dimensions.

        Returns
        -------
        int
            The number of axes.
        """
        return len(self.coordinates)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the grid shape.

        Returns
        -------
        tuple[int, ...]
            The number of points along each axis.
        """
        return tuple(axis.size for axis in self.coordinates)

    @property
    def num_points(self) -> int:
        """
        Get the total number of grid points.

        Returns
        -------
        int
            The product of the grid shape.
        """
        return prod(self.shape)

    @property
    def spacings(self) -> FloatArray:
        """
        Get the distance between neighboring points along each axis.

        Returns
        -------
        FloatArray
            The spacing of each axis with shape (num_dimensions,).
        """
        return np.array(
            [(axis[-1] - axis[0]) / (axis.size - 1) for axis in self.coordinates],
            dtype=np.float64,
        )

    @property
    def cell_volume(self) -> float:
        """
        Get the volume of one grid cell.

        Returns
        -------
        float
            The product of the spacings.
        """
        return float(np.prod(self.spacings))

    @property
    def points(self) -> FloatArray:
        """
        Get every grid point as a sample matrix.

        Returns
        -------
        FloatArray
            The points with shape (num_points, num_dimensions), in the C order
            of a grid with ``shape``.
        """
        mesh = np.meshgrid(*self.coordinates, indexing="ij")
        return np.stack([component.ravel() for component in mesh], axis=1)

    def without_axis(self, axis: int) -> GridAxes:
        """
        Get the axes with one dimension removed.

        Parameters
        ----------
        axis : int
            The index of the dimension to remove.

        Returns
        -------
        GridAxes
            The remaining axes in their original order.

        Raises
        ------
        ValueError
            If only one dimension remains.
        IndexError
            If axis is outside the range [0, num_dimensions).
        """
        if not 0 <= axis < self.num_dimensions:
            raise IndexError(
                f"axis: {axis} is out of range for {self.num_dimensions} dimensions"
            )
        if self.num_dimensions == 1:
            raise ValueError("cannot remove the only axis")
        return GridAxes(
            [coords for index, coords in enumerate(self.coordinates) if index != axis]
        )
