"""
Geradores de dados sintéticos
"""

from datatunner.generators.base import BaseSyntheticGenerator
from datatunner.generators.augmentation import ImageAugmentation
from datatunner.generators.smote import SMOTEGenerator, GaussianNoiseGenerator

# Importações condicionais para evitar erros se libs não instaladas
try:
    from datatunner.generators.ctgan import CTGANGenerator, TVAEGenerator
except ImportError:
    CTGANGenerator = None
    TVAEGenerator = None

try:
    from datatunner.generators.diffusion import StableDiffusionGenerator, DreamBoothGenerator
except ImportError:
    StableDiffusionGenerator = None
    DreamBoothGenerator = None

__all__ = [
    'BaseSyntheticGenerator',
    'ImageAugmentation',
    'SMOTEGenerator',
    'GaussianNoiseGenerator',
    'CTGANGenerator',
    'TVAEGenerator',
    'StableDiffusionGenerator',
    'DreamBoothGenerator'
]
