from enum import Enum


class MixtureCovarianceType(Enum):
    """
    Covariance structure of the components of a Gaussian mixture.

    Attributes
    ----------
    FULL
        Each component has its own general covariance matrix.
    TIED
        All components share one general covariance matrix.
    DIAG
        Each component has its own diagonal covariance matrix.
    SPHERICAL
        Each component has its own single variance.
    """

    FULL = "full"
    TIED = "tied"
    DIAG = "diag"
    SPHERICAL = "spherical"

    def get_num_covariance_parameters(
        self, num_components: int, num_features: int
    ) -> int:
        """
        Get the number of free covariance parameters of a mixture.

        Parameters
        ----------
        num_components : int
            The number of mixture components.
        num_features : int
            The number of features.

        Returns
        -------
        int
            The number of free covariance parameters.
        """
        symmetric_matrix_size = num_features * (num_features + 1) // 2
        match self:
            case MixtureCovarianceType.FULL:
                return num_components * symmetric_matrix_size
            case MixtureCovarianceType.TIED:
                return symmetric_matrix_size
            case MixtureCovarianceType.DIAG:
                return num_components * num_features
            case MixtureCovarianceType.SPHERICAL:
                return num_components
