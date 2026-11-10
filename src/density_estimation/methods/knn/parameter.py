from dataclasses import dataclass
from typing import override

from ..parameter import DensityParameters
from .metric import KNNMetric
from .processor import KNNProcessor


@dataclass(frozen=True)
class KNNParameters(DensityParameters):
    """
    Parameters for k-nearest neighbor density estimation.

    Attributes
    ----------
    num_neighbors : int
        The neighbor rank k whose distance sets the density. Larger values
        give smoother estimates.
    metric : KNNMetric
        The distance metric.
    """

    num_neighbors: int = 10
    metric: KNNMetric = KNNMetric.EUCLIDEAN

    def __post_init__(self) -> None:
        if self.num_neighbors <= 0:
            raise ValueError("num_neighbors must be greater than 0")

    @override
    def build_processor(self) -> KNNProcessor:
        return KNNProcessor(
            num_neighbors=self.num_neighbors,
            metric=self.metric,
        )
