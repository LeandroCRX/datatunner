"""
Classe base abstrata para geradores de dados sintéticos
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional


class BaseSyntheticGenerator(ABC):
    """Classe base para geradores de dados sintéticos"""
    
    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: Seed para reprodutibilidade
        """
        self.random_seed = random_seed
        self.generator_name = "BaseGenerator"
        np.random.seed(random_seed)
    
    @abstractmethod
    def fit(self, data: Any, labels: Optional[np.ndarray] = None):
        """
        Ajusta o gerador aos dados reais
        
        Args:
            data: Dados reais
            labels: Labels dos dados (se aplicável)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        n_samples: int,
        **kwargs
    ) -> Any:
        """
        Gera dados sintéticos
        
        Args:
            n_samples: Número de amostras a gerar
            **kwargs: Argumentos adicionais
            
        Returns:
            Dados sintéticos gerados
        """
        pass
    
    def get_generator_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o gerador
        
        Returns:
            Dicionário com informações
        """
        return {
            "generator_name": self.generator_name,
            "random_seed": self.random_seed
        }
