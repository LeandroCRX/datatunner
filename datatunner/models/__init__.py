"""
Modelos de Machine Learning
"""

from datatunner.models.base import BaseModel
from datatunner.models.cnn import (
    ResNetClassifier,
    VGGClassifier,
    MobileNetClassifier,
    CustomCNN
)
from datatunner.models.mlp import (
    MLPClassifier,
    MLPRegressor,
    DeepMLP
)

# Importação condicional de modelos clássicos
try:
    from datatunner.models.classical import (
        DecisionTreeClassifier,
        RandomForestClassifier,
        XGBoostClassifier,
        LightGBMClassifier,
        CatBoostClassifier,
        SVMClassifier,
        LogisticRegressionClassifier,
        NaiveBayesClassifier,
        KNNClassifier
    )
except ImportError:
    # Fallback se alguma biblioteca não estiver instalada
    DecisionTreeClassifier = None
    RandomForestClassifier = None
    XGBoostClassifier = None
    LightGBMClassifier = None
    CatBoostClassifier = None
    SVMClassifier = None
    LogisticRegressionClassifier = None
    NaiveBayesClassifier = None
    KNNClassifier = None

__all__ = [
    # Base
    'BaseModel',
    # CNNs
    'ResNetClassifier',
    'VGGClassifier',
    'MobileNetClassifier',
    'CustomCNN',
    # MLPs
    'MLPClassifier',
    'MLPRegressor',
    'DeepMLP',
    # Classical ML
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'XGBoostClassifier',
    'LightGBMClassifier',
    'CatBoostClassifier',
    'SVMClassifier',
    'LogisticRegressionClassifier',
    'NaiveBayesClassifier',
    'KNNClassifier'
]
