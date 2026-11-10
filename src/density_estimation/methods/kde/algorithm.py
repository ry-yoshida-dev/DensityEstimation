from enum import Enum


class KDEAlgorithm(Enum):
    """
    Tree used by scikit-learn's KernelDensity to accelerate evaluation.

    Attributes
    ----------
    AUTO
        Let scikit-learn choose.
    KD_TREE
        KD-tree, fast in low dimensions.
    BALL_TREE
        Ball tree, more robust in higher dimensions.
    """

    AUTO = "auto"
    KD_TREE = "kd_tree"
    BALL_TREE = "ball_tree"
