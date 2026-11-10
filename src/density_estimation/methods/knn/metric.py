import math
from enum import Enum


class KNNMetric(Enum):
    """
    Distance metric of k-NN density estimation.

    Each metric defines its own unit ball, whose volume normalizes the density.

    Attributes
    ----------
    EUCLIDEAN
        L2 distance; the unit ball is a sphere.
    MANHATTAN
        L1 distance; the unit ball is a cross-polytope.
    CHEBYSHEV
        L-infinity distance; the unit ball is a hypercube.
    """

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"

    def get_log_unit_ball_volume(self, num_dimensions: int) -> float:
        """
        Get the log volume of the unit ball of this metric.

        Parameters
        ----------
        num_dimensions : int
            The number of dimensions, at least 1.

        Returns
        -------
        float
            The natural logarithm of the unit ball volume.

        Raises
        ------
        ValueError
            If num_dimensions is below 1.
        """
        if num_dimensions < 1:
            raise ValueError(f"num_dimensions must be at least 1, got {num_dimensions}")
        match self:
            case KNNMetric.EUCLIDEAN:
                return 0.5 * num_dimensions * math.log(math.pi) - math.lgamma(
                    0.5 * num_dimensions + 1.0
                )
            case KNNMetric.MANHATTAN:
                return num_dimensions * math.log(2.0) - math.lgamma(
                    num_dimensions + 1.0
                )
            case KNNMetric.CHEBYSHEV:
                return num_dimensions * math.log(2.0)
