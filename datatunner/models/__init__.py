"""
Modelos de Machine Learning

Modelos PyTorch (integraveis com DataTunner.optimize()):
  - CNNs: ResNetClassifier, VGGClassifier, MobileNetClassifier, CustomCNN
  - MLPs: MLPClassifier, MLPRegressor, DeepMLP

Modelos clássicos sklearn (utilitarios autonomos, NAO integraveis
diretamente com DataTunner.optimize() que exige nn.Module):
  - DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost, SVM,
    LogisticRegression, NaiveBayes, KNN
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

# Importacao condicional de modelos classicos (sklearn)
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
