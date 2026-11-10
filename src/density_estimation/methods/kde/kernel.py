from enum import Enum


class KDEKernel(Enum):
    """
    Kernel of scikit-learn's KernelDensity.

    Attributes
    ----------
    GAUSSIAN
        Gaussian kernel with unbounded support.
    TOPHAT
        Uniform kernel on a ball of radius bandwidth.
    EPANECHNIKOV
        Parabolic kernel on a ball of radius bandwidth.
    EXPONENTIAL
        Exponential kernel with unbounded support.
    LINEAR
        Triangular kernel on a ball of radius bandwidth.
    COSINE
        Cosine kernel on a ball of radius bandwidth.
    """

    GAUSSIAN = "gaussian"
    TOPHAT = "tophat"
    EPANECHNIKOV = "epanechnikov"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    COSINE = "cosine"

    @property
    def is_sampling_supported(self) -> bool:
        """
        Check whether scikit-learn can draw samples with this kernel.

        Returns
        -------
        bool
            True for the Gaussian and top-hat kernels.
        """
        match self:
            case KDEKernel.GAUSSIAN | KDEKernel.TOPHAT:
                return True
            case (
                KDEKernel.EPANECHNIKOV
                | KDEKernel.EXPONENTIAL
                | KDEKernel.LINEAR
                | KDEKernel.COSINE
            ):
                return False
