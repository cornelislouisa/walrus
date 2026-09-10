try:
    from .baseline_wrappers import DPOTWrapper, MPPWrapper, ScOTWrapper
except ImportError:
    # Allow for partial installs without these dependencies
    DPOTWrapper = None
    ScOTWrapper = None
    MPPWrapper = None
from walrus.baselines import (
    AdvectionWrapper,
    FFNOWrapper,
    MeanFieldWrapper,
    SineNetWrapper,
)

from .isotropic_model import IsotropicModel, IsotropicModelWithNoise

__all__ = [
    "IsotropicModel",
    "IsotropicModelWithNoise",
    "DPOTWrapper",
    "ScOTWrapper",
    "MPPWrapper",
    "AdvectionWrapper",
    "FFNOWrapper",
    "MeanFieldWrapper",
    "SineNetWrapper",
]
