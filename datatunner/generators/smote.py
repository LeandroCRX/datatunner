"""
SMOTE para geração de dados tabulares sintéticos
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from typing import Union, Optional, Tuple

from datatunner.generators.base import BaseSyntheticGenerator


class SMOTEGenerator(BaseSyntheticGenerator):
    """Gerador de dados tabulares usando SMOTE"""
    
    def __init__(
        self,
        k_neighbors: int = 5,
        random_seed: int = 42,
        variant: str = "standard"
    ):
        """
        Args:
            k_neighbors: Número de vizinhos para SMOTE
            random_seed: Seed para reprodutibilidade
            variant: Variante do SMOTE (standard, borderline, adasyn)
        """
        super().__init__(random_seed)
        
        self.generator_name = f"SMOTE-{variant}"
        self.k_neighbors = k_neighbors
        self.variant = variant
        
        # Criar gerador apropriado
        if variant == "standard":
            self.smote = SMOTE(
                k_neighbors=k_neighbors,
                random_state=random_seed
            )
        elif variant == "borderline":
            self.smote = BorderlineSMOTE(
                k_neighbors=k_neighbors,
                random_state=random_seed
            )
        elif variant == "adasyn":
            self.smote = ADASYN(
                n_neighbors=k_neighbors,
                random_state=random_seed
            )
        else:
            raise ValueError(f"Variante inválida: {variant}")
        
        self.X_train = None
        self.y_train = None
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: np.ndarray
    ):
        """
        Ajusta SMOTE aos dados
        
        Args:
            data: Features de treino
            labels: Labels de treino
        """
        if isinstance(data, pd.DataFrame):
            self.X_train = data.values
        else:
            self.X_train = data
        
        self.y_train = labels
    
    def generate(
        self,
        n_samples: int = None,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera dados sintéticos usando SMOTE
        
        Args:
            n_samples: Número de amostras (ignorado para SMOTE automático)
            target_class: Classe alvo (se None, balanceia todas)
            
        Returns:
            Tupla (X_synthetic, y_synthetic)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Execute fit() antes de generate()")
        
        # SMOTE gera automaticamente para balancear
        X_resampled, y_resampled = self.smote.fit_resample(
            self.X_train, self.y_train
        )
        
        # Extrair apenas os sintéticos (depois dos reais)
        n_original = len(self.X_train)
        X_synthetic = X_resampled[n_original:]
        y_synthetic = y_resampled[n_original:]
        
        # Se n_samples especificado, amostrar
        if n_samples and n_samples < len(X_synthetic):
            indices = np.random.choice(len(X_synthetic), n_samples, replace=False)
            X_synthetic = X_synthetic[indices]
            y_synthetic = y_synthetic[indices]
        
        return X_synthetic, y_synthetic
    
    def get_generator_info(self):
        info = super().get_generator_info()
        info['k_neighbors'] = self.k_neighbors
        info['variant'] = self.variant
        return info


class GaussianNoiseGenerator(BaseSyntheticGenerator):
    """Gerador simples usando ruído Gaussiano"""
    
    def __init__(
        self,
        noise_level: float = 0.1,
        random_seed: int = 42
    ):
        """
        Args:
            noise_level: Nível de ruído (desvio padrão relativo)
            random_seed: Seed
        """
        super().__init__(random_seed)
        
        self.generator_name = "GaussianNoise"
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: np.ndarray
    ):
        """Armazena dados de treino"""
        if isinstance(data, pd.DataFrame):
            self.X_train = data.values
        else:
            self.X_train = data
        
        self.y_train = labels
    
    def generate(
        self,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera dados sintéticos adicionando ruído Gaussiano
        
        Args:
            n_samples: Número de amostras
            
        Returns:
            Tupla (X_synthetic, y_synthetic)
        """
        if self.X_train is None:
            raise ValueError("Execute fit() antes de generate()")
        
        # Selecionar amostras base aleatórias
        indices = np.random.choice(len(self.X_train), n_samples, replace=True)
        X_base = self.X_train[indices]
        y_synthetic = self.y_train[indices]
        
        # Adicionar ruído Gaussiano
        std = np.std(self.X_train, axis=0) * self.noise_level
        noise = np.random.normal(0, std, size=X_base.shape)
        X_synthetic = X_base + noise
        
        return X_synthetic, y_synthetic
