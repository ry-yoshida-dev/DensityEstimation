from enum import Enum


class DensityEstimationMethod(Enum):
    """
    Density estimation method implemented by a processor.

    Attributes
    ----------
    KDE
        Kernel density estimation.
    GMM
        Gaussian mixture model fitted by expectation-maximization.
    BAYESIAN_GMM
        Gaussian mixture model fitted by variational inference.
    KNN
        k-nearest neighbor density estimation.
    """

    KDE = "KDE"
    GMM = "GMM"
    BAYESIAN_GMM = "BayesianGMM"
    KNN = "kNN"
