"""
Classe base abstrata para modelos
"""

from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """Classe base para todos os modelos do DataTunner"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "BaseModel"
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo
        
        Returns:
            Dicionário com informações
        """
        pass
    
    def count_parameters(self) -> int:
        """
        Conta número de parâmetros treináveis
        
        Returns:
            Número de parâmetros
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_layers(self, num_layers: int = 0):
        """
        Congela camadas para transfer learning
        
        Args:
            num_layers: Número de camadas a congelar (0 = todas)
        """
        if num_layers == 0:
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Implementar lógica específica por modelo
            pass
    
    def unfreeze_all(self):
        """Descongela todas as camadas"""
        for param in self.parameters():
            param.requires_grad = True
