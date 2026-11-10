from typing import Literal

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray

class GaussianMixture:
    n_components: int
    covariance_type: Literal["full", "tied", "diag", "spherical"]
    tol: float
    reg_covar: float
    max_iter: int
    n_init: int
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"]
    weights_init: ArrayLike | None
    means_init: ArrayLike | None
    precisions_init: ArrayLike | None
    random_state: int | RandomState | None
    warm_start: bool
    verbose: int
    verbose_interval: int

    weights_: NDArray[np.floating]
    means_: NDArray[np.floating]
    covariances_: NDArray[np.floating]
    precisions_: NDArray[np.floating]
    precisions_cholesky_: NDArray[np.floating]
    converged_: bool
    n_iter_: int
    lower_bound_: float
    n_features_in_: int

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = "kmeans",
        weights_init: ArrayLike | None = None,
        means_init: ArrayLike | None = None,
        precisions_init: ArrayLike | None = None,
        random_state: int | RandomState | None = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> GaussianMixture: ...
    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.integer]: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]: ...
    def sample(
        self, n_samples: int = 1
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]: ...
    def score(self, X: ArrayLike, y: ArrayLike | None = None) -> float: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]: ...
    def bic(self, X: ArrayLike) -> float: ...
    def aic(self, X: ArrayLike) -> float: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> GaussianMixture: ...

class BayesianGaussianMixture:
    n_components: int
    covariance_type: Literal["full", "tied", "diag", "spherical"]
    tol: float
    reg_covar: float
    max_iter: int
    n_init: int
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"]
    weights_init: ArrayLike | None
    means_init: ArrayLike | None
    precisions_init: ArrayLike | None
    weight_concentration_prior_type: Literal["dirichlet_process", "dirichlet_distribution"]
    weight_concentration_prior: float | None
    mean_precision_prior: float | None
    mean_prior: ArrayLike | None
    degrees_of_freedom_prior: float | None
    covariance_prior: float | ArrayLike | None
    random_state: int | RandomState | None
    warm_start: bool
    verbose: int
    verbose_interval: int

    weights_: NDArray[np.floating]
    means_: NDArray[np.floating]
    covariances_: NDArray[np.floating]
    precisions_: NDArray[np.floating]
    precisions_cholesky_: NDArray[np.floating]
    converged_: bool
    n_iter_: int
    lower_bound_: float
    weight_concentration_: tuple[NDArray[np.floating], NDArray[np.floating]] | NDArray[np.floating]
    n_features_in_: int

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = "kmeans",
        weights_init: ArrayLike | None = None,
        means_init: ArrayLike | None = None,
        precisions_init: ArrayLike | None = None,
        weight_concentration_prior_type: Literal["dirichlet_process", "dirichlet_distribution"] = "dirichlet_process",
        weight_concentration_prior: float | None = None,
        mean_precision_prior: float | None = None,
        mean_prior: ArrayLike | None = None,
        degrees_of_freedom_prior: float | None = None,
        covariance_prior: float | ArrayLike | None = None,
        random_state: int | RandomState | None = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BayesianGaussianMixture: ...
    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.integer]: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]: ...
    def sample(
        self, n_samples: int = 1
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]: ...
    def score(self, X: ArrayLike, y: ArrayLike | None = None) -> float: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> BayesianGaussianMixture: ...
