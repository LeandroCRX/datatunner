"""
Modelos clássicos de Machine Learning
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.svm import SVC as SKSVC
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.naive_bayes import GaussianNB as SKGaussianNB
from sklearn.neighbors import KNeighborsClassifier as SKKNeighbors

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ClassicalMLModel:
    """Classe base para modelos clássicos de ML"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.model_name = "ClassicalML"
        self.is_fitted = False
    
    def fit(self, X, y):
        """Treina o modelo"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Faz predições"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Retorna probabilidades"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado")
        return self.model.predict_proba(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        return {
            "model_name": self.model_name,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted
        }


class DecisionTreeClassifier(ClassicalMLModel):
    """
    Decision Tree Classifier
    
    Árvore de decisão para classificação
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        random_state: int = 42
    ):
        """
        Args:
            max_depth: Profundidade máxima da árvore
            min_samples_split: Mínimo de amostras para split
            min_samples_leaf: Mínimo de amostras por folha
            criterion: Critério de split ('gini' ou 'entropy')
            random_state: Seed
        """
        super().__init__(random_state)
        
        self.model_name = "DecisionTree"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        
        self.model = SKDecisionTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Retorna importância das features"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado")
        return self.model.feature_importances_


class RandomForestClassifier(ClassicalMLModel):
    """
    Random Forest Classifier
    
    Ensemble de árvores de decisão
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Args:
            n_estimators: Número de árvores
            max_depth: Profundidade máxima
            min_samples_split: Mínimo de amostras para split
            min_samples_leaf: Mínimo de amostras por folha
            criterion: Critério ('gini' ou 'entropy')
            max_features: Número de features por split
            n_jobs: Paralelização (-1 = todos os cores)
            random_state: Seed
        """
        super().__init__(random_state)
        
        self.model_name = "RandomForest"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.model = SKRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Retorna importância das features"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado")
        return self.model.feature_importances_


class XGBoostClassifier(ClassicalMLModel):
    """
    XGBoost Classifier
    
    Gradient Boosting otimizado
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False
    ):
        """
        Args:
            n_estimators: Número de árvores
            max_depth: Profundidade máxima
            learning_rate: Taxa de aprendizado
            subsample: Fração de amostras por árvore
            colsample_bytree: Fração de features por árvore
            gamma: Regularização
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Seed
            n_jobs: Paralelização
            use_gpu: Se deve usar GPU
        """
        super().__init__(random_state)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost não instalado. Instale com: pip install xgboost"
            )
        
        self.model_name = "XGBoost"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        tree_method = 'hist'

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method=tree_method,
            device='cuda' if use_gpu else 'cpu',
            eval_metric='logloss'
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Retorna importância das features"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado")
        return self.model.feature_importances_


class LightGBMClassifier(ClassicalMLModel):
    """
    LightGBM Classifier
    
    Gradient Boosting rápido e eficiente
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        random_state: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False
    ):
        """
        Args:
            n_estimators: Número de árvores
            max_depth: Profundidade máxima (-1 = sem limite)
            learning_rate: Taxa de aprendizado
            num_leaves: Número máximo de folhas
            subsample: Fração de amostras
            colsample_bytree: Fração de features
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Seed
            n_jobs: Paralelização
            use_gpu: Se deve usar GPU
        """
        super().__init__(random_state)
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM não instalado. Instale com: pip install lightgbm"
            )
        
        self.model_name = "LightGBM"
        
        device = 'gpu' if use_gpu else 'cpu'
        
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            verbose=-1
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Retorna importância das features"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado")
        return self.model.feature_importances_


class CatBoostClassifier(ClassicalMLModel):
    """
    CatBoost Classifier
    
    Gradient Boosting com suporte nativo a variáveis categóricas
    """
    
    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        verbose: bool = False,
        use_gpu: bool = False
    ):
        """
        Args:
            iterations: Número de árvores
            learning_rate: Taxa de aprendizado
            depth: Profundidade
            l2_leaf_reg: L2 regularization
            random_state: Seed
            verbose: Verbosidade
            use_gpu: Se deve usar GPU
        """
        super().__init__(random_state)
        
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost não instalado. Instale com: pip install catboost"
            )
        
        self.model_name = "CatBoost"
        
        task_type = 'GPU' if use_gpu else 'CPU'
        
        self.model = cb.CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_state=random_state,
            verbose=verbose,
            task_type=task_type
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Retorna importância das features"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado")
        return self.model.feature_importances_


class SVMClassifier(ClassicalMLModel):
    """
    Support Vector Machine Classifier
    
    SVM com diferentes kernels
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        random_state: int = 42
    ):
        """
        Args:
            kernel: Tipo de kernel ('linear', 'rbf', 'poly', 'sigmoid')
            C: Parâmetro de regularização
            gamma: Coeficiente do kernel
            random_state: Seed
        """
        super().__init__(random_state)
        
        self.model_name = f"SVM-{kernel}"
        self.kernel = kernel
        self.C = C
        
        self.model = SKSVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True  # Para predict_proba
        )


class LogisticRegressionClassifier(ClassicalMLModel):
    """
    Logistic Regression
    
    Regressão logística para classificação
    """
    
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 100,
        random_state: int = 42
    ):
        """
        Args:
            C: Inverso da força de regularização
            penalty: Tipo de regularização ('l1', 'l2', 'elasticnet')
            solver: Algoritmo de otimização
            max_iter: Número máximo de iterações
            random_state: Seed
        """
        super().__init__(random_state)
        
        self.model_name = "LogisticRegression"
        
        self.model = SKLogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )


class NaiveBayesClassifier(ClassicalMLModel):
    """
    Naive Bayes Classifier
    
    Classificador Bayesiano
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Args:
            var_smoothing: Smoothing parameter
        """
        super().__init__(random_state=42)
        
        self.model_name = "NaiveBayes"
        
        self.model = SKGaussianNB(var_smoothing=var_smoothing)


class KNNClassifier(ClassicalMLModel):
    """
    K-Nearest Neighbors Classifier
    
    Classificador baseado em vizinhos mais próximos
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski",
        n_jobs: int = -1
    ):
        """
        Args:
            n_neighbors: Número de vizinhos
            weights: Tipo de peso ('uniform' ou 'distance')
            metric: Métrica de distância
            n_jobs: Paralelização
        """
        super().__init__(random_state=42)
        
        self.model_name = "KNN"
        self.n_neighbors = n_neighbors
        
        self.model = SKKNeighbors(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=n_jobs
        )
