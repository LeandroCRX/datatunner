"""
Geradores de dados sinteticos

Imagens:
  - ImageAugmentation: data augmentation (albumentations)

Dados tabulares:
  - SMOTEGenerator: SMOTE e variantes (standard, borderline, adasyn)
  - GaussianNoiseGenerator: ruido gaussiano
  - CTGANGenerator: Conditional Tabular GAN (SDV)
  - TVAEGenerator: Tabular VAE (SDV)
"""

from datatunner.generators.base import BaseSyntheticGenerator
from datatunner.generators.augmentation import ImageAugmentation
from datatunner.generators.smote import SMOTEGenerator, GaussianNoiseGenerator

try:
    from datatunner.generators.ctgan import CTGANGenerator, TVAEGenerator
except ImportError:
    CTGANGenerator = None
    TVAEGenerator = None

__all__ = [
    'BaseSyntheticGenerator',
    'ImageAugmentation',
    'SMOTEGenerator',
    'GaussianNoiseGenerator',
    'CTGANGenerator',
    'TVAEGenerator',
]
