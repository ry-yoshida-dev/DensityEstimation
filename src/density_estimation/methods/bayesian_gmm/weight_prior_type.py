from enum import Enum


class BayesianGMMWeightPriorType(Enum):
    """
    Prior on the component weights of a Bayesian Gaussian mixture.

    Attributes
    ----------
    DIRICHLET_PROCESS
        Stick-breaking Dirichlet process prior, which concentrates weight on
        few components and lets the rest vanish.
    DIRICHLET_DISTRIBUTION
        Finite Dirichlet distribution prior, which spreads weight more evenly.
    """

    DIRICHLET_PROCESS = "dirichlet_process"
    DIRICHLET_DISTRIBUTION = "dirichlet_distribution"
