"""External baseline models wrapped for the Walrus training API."""

from .advection import AdvectionWrapper
from .ffno import FFNOWrapper
from .mean_field import MeanFieldWrapper
from .sinenet import SineNetWrapper

__all__ = [
    "AdvectionWrapper",
    "FFNOWrapper",
    "MeanFieldWrapper",
    "SineNetWrapper",
]
