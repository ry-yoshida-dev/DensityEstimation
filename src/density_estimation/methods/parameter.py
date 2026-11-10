from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import DensityProcessor


class DensityParameters(ABC):
    """
    Base class for density estimation parameters that build a processor.
    """

    @abstractmethod
    def build_processor(self) -> DensityProcessor:
        """
        Build the density processor.

        Returns
        -------
        DensityProcessor
            The density processor.
        """
