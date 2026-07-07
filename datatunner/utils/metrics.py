"""
Cálculo de métricas de avaliação
"""

import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class MetricsCalculator:
    """Calculadora de métricas para classificação e regressão"""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Calcula métricas de classificação
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            y_pred_proba: Probabilidades das predições (para ROC-AUC)
            average: Tipo de média para métricas multiclasse
            
        Returns:
            Dicionário com métricas
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        
        # ROC-AUC apenas se as probabilidades forem fornecidas
        if y_pred_proba is not None:
            try:
                # Para classificação binária
                if y_pred_proba.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                # Para classificação multiclasse
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_pred_proba, multi_class="ovr", average=average
                    )
            except Exception as e:
                print(f"Aviso: Não foi possível calcular ROC-AUC: {e}")
                metrics["roc_auc"] = None
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula métricas de regressão
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            
        Returns:
            Dicionário com métricas
        """
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calcula matriz de confusão
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            
        Returns:
            Matriz de confusão
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Gera relatório de classificação detalhado
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            target_names: Nomes das classes
            
        Returns:
            Relatório em formato string
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    @staticmethod
    def calculate_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula métricas por classe
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            class_names: Nomes das classes
            
        Returns:
            Dicionário com métricas por classe
        """
        unique_classes = np.unique(y_true)
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_classes]
        
        per_class_metrics = {}
        
        for idx, class_id in enumerate(unique_classes):
            class_mask = (y_true == class_id)
            
            per_class_metrics[class_names[idx]] = {
                "precision": precision_score(
                    y_true == class_id, y_pred == class_id, zero_division=0
                ),
                "recall": recall_score(
                    y_true == class_id, y_pred == class_id, zero_division=0
                ),
                "f1_score": f1_score(
                    y_true == class_id, y_pred == class_id, zero_division=0
                ),
                "support": int(class_mask.sum()),
            }
        
        return per_class_metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "classification",
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Função de conveniência para calcular métricas
    
    Args:
        y_true: Labels ou valores verdadeiros
        y_pred: Predições
        task_type: "classification" ou "regression"
        y_pred_proba: Probabilidades (apenas para classificação)
        class_names: Nomes das classes (apenas para classificação)
        
    Returns:
        Dicionário com todas as métricas
    """
    calculator = MetricsCalculator()
    
    if task_type == "classification":
        metrics = calculator.calculate_classification_metrics(
            y_true, y_pred, y_pred_proba
        )
        metrics["confusion_matrix"] = calculator.get_confusion_matrix(
            y_true, y_pred
        ).tolist()
        metrics["per_class"] = calculator.calculate_per_class_metrics(
            y_true, y_pred, class_names
        )
    elif task_type == "regression":
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Tipo de tarefa inválido: {task_type}")
    
    return metrics
