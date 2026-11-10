from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ..grid_axes import GridAxes
from ..types import BoolArray, FloatArray


@dataclass(frozen=True, eq=False)
class DensityGrid:
    """
    Immutable log densities evaluated on a rectangular grid.

    Integrals over the grid are Riemann sums over its cells, so they only
    approximate the true integrals and miss any mass outside the grid.

    Attributes
    ----------
    axes : GridAxes
        The grid coordinates.
    log_densities : FloatArray
        The log densities as a read-only array with shape ``axes.shape``
        owned by this instance.
    """

    axes: GridAxes
    log_densities: FloatArray = field(init=False, repr=False)

    def __init__(self, axes: GridAxes, log_densities: ArrayLike) -> None:
        owned = np.array(log_densities, dtype=np.float64)
        if owned.size != axes.num_points:
            raise ValueError(
                f"log_densities must have {axes.num_points} values for grid shape {axes.shape}, got {owned.size}"
            )
        if np.any(np.isnan(owned)):
            raise ValueError("log_densities must not contain NaN")
        owned = owned.reshape(axes.shape)
        owned.setflags(write=False)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "log_densities", owned)

    @property
    def densities(self) -> FloatArray:
        """
        Get the densities.

        Returns
        -------
        FloatArray
            The exponentiated log densities with shape ``axes.shape``.
        """
        return np.exp(self.log_densities)

    @property
    def probability_mass(self) -> float:
        """
        Get the probability mass captured by the grid.

        Close to one when the grid covers the support of a normalized density
        finely enough, which makes it a check of grid coverage.

        Returns
        -------
        float
            The Riemann sum of the densities.
        """
        return float(np.sum(self.densities) * self.axes.cell_volume)

    @property
    def max_log_density(self) -> float:
        """
        Get the highest log density on the grid.

        Returns
        -------
        float
            The maximum log density.
        """
        return float(np.max(self.log_densities))

    @property
    def mode(self) -> FloatArray:
        """
        Get the grid point with the highest density.

        Returns
        -------
        FloatArray
            The coordinates of the first maximum in C order, with shape
            (num_dimensions,).
        """
        index = np.unravel_index(np.argmax(self.log_densities), self.axes.shape)
        return np.array(
            [
                axis[position]
                for axis, position in zip(self.axes.coordinates, index, strict=True)
            ],
            dtype=np.float64,
        )

    def get_highest_density_threshold(self, probability: float) -> float:
        """
        Get the log-density level of a highest density region.

        The highest density region for ``probability`` is the smallest set of
        grid cells whose densities all exceed the level and whose mass is at
        least ``probability`` of the grid's mass.

        Parameters
        ----------
        probability : float
            The fraction of the grid's mass in the range (0, 1].

        Returns
        -------
        float
            The log density of the lowest cell inside the region.

        Raises
        ------
        ValueError
            If probability is outside the range (0, 1], or the maximum log
            density is not finite.
        """
        if not 0.0 < probability <= 1.0:
            raise ValueError(f"probability must be in (0, 1], got {probability}")
        max_log_density = self.max_log_density
        if not np.isfinite(max_log_density):
            raise ValueError(
                f"highest density regions are undefined when the maximum log density is {max_log_density}"
            )
        descending = np.sort(self.log_densities, axis=None)[::-1]
        # The cumulative sum saturates at 1 before the far tail, so the whole
        # grid must be selected explicitly.
        if probability == 1.0:
            lowest: float = descending.item(-1)
            return lowest
        relative_densities: FloatArray = np.exp(descending - max_log_density)
        cumulative = np.cumsum(relative_densities)
        cumulative /= cumulative.item(-1)
        position = min(
            int(np.searchsorted(cumulative, probability, side="left")),
            descending.size - 1,
        )
        level: float = descending.item(position)
        return level

    def get_highest_density_region_mask(self, probability: float) -> BoolArray:
        """
        Get the cells of a highest density region.

        Parameters
        ----------
        probability : float
            The fraction of the grid's mass in the range (0, 1].

        Returns
        -------
        BoolArray
            The mask with shape ``axes.shape``, True inside the region.

        Raises
        ------
        ValueError
            If probability is outside the range (0, 1], or the maximum log
            density is not finite.
        """
        threshold = self.get_highest_density_threshold(probability)
        return np.greater_equal(self.log_densities, threshold)

    def marginalize(self, axis: int) -> DensityGrid:
        """
        Integrate out one dimension.

        Parameters
        ----------
        axis : int
            The index of the dimension to integrate out.

        Returns
        -------
        DensityGrid
            The marginal log densities over the remaining dimensions.

        Raises
        ------
        ValueError
            If the grid has only one dimension.
        IndexError
            If axis is outside the range [0, num_dimensions).
        """
        remaining_axes = self.axes.without_axis(axis)
        shift = np.max(self.log_densities, axis=axis, keepdims=True)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        with np.errstate(divide="ignore"):
            summed = np.log(
                np.sum(np.exp(self.log_densities - shift), axis=axis, keepdims=True)
            )
        log_spacing = math.log(self.axes.spacings.item(axis))
        marginal: FloatArray = np.squeeze(summed + shift, axis=axis) + log_spacing
        return DensityGrid(remaining_axes, marginal)
