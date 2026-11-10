from typing import Literal, overload

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray

class KernelDensity:
    bandwidth: float | Literal["scott", "silverman"]
    algorithm: Literal["auto", "kd_tree", "ball_tree"]
    kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    metric: str
    atol: float
    rtol: float
    breadth_first: bool
    leaf_size: int
    metric_params: dict[str, object] | None

    bandwidth_: float
    n_features_in_: int

    def __init__(
        self,
        *,
        bandwidth: float | Literal["scott", "silverman"] = 1.0,
        algorithm: Literal["auto", "kd_tree", "ball_tree"] = "auto",
        kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
        metric: str = "euclidean",
        atol: float = 0,
        rtol: float = 0,
        breadth_first: bool = True,
        leaf_size: int = 40,
        metric_params: dict[str, object] | None = None,
    ) -> None: ...
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> KernelDensity: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def score(self, X: ArrayLike, y: ArrayLike | None = None) -> float: ...
    def sample(
        self, n_samples: int = 1, random_state: int | RandomState | None = None
    ) -> NDArray[np.float64]: ...

class NearestNeighbors:
    n_neighbors: int
    radius: float
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"]
    leaf_size: int
    metric: str
    p: float
    metric_params: dict[str, object] | None
    n_jobs: int | None

    n_features_in_: int
    n_samples_fit_: int

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: float = 2,
        metric_params: dict[str, object] | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> NearestNeighbors: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike | None = None,
        n_neighbors: int | None = None,
        return_distance: Literal[True] = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike | None = None,
        n_neighbors: int | None = None,
        *,
        return_distance: Literal[False],
    ) -> NDArray[np.intp]: ...
