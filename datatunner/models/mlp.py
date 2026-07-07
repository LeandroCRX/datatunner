"""
Modelos de Perceptrons de Múltiplas Camadas (MLP)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from datatunner.models.base import BaseModel


class MLPClassifier(BaseModel):
    """Classificador MLP para dados tabulares"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.3,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Dimensao de entrada
            num_classes: Numero de classes
            hidden_layers: Lista com numero de neuronios por camada oculta
            dropout: Taxa de dropout
            batch_norm: Se deve usar batch normalization
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.model_name = "MLP"
        
        # Construir camadas
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Camada de saída
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_layers": self.hidden_layers,
            "num_parameters": self.count_parameters()
        }


class MLPRegressor(BaseModel):
    """Regressor MLP para dados tabulares"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.3,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Dimensao de entrada
            output_dim: Dimensao de saida
            hidden_layers: Lista com numero de neuronios por camada oculta
            dropout: Taxa de dropout
            batch_norm: Se deve usar batch normalization
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.model_name = "MLPRegressor"
        
        # Construir camadas
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Camada de saída (sem ativação para regressão)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "num_parameters": self.count_parameters()
        }


class DeepMLP(BaseModel):
    """MLP profundo com residual connections"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_residual: bool = True
    ):
        """
        Args:
            input_dim: Dimensao de entrada
            num_classes: Numero de classes
            hidden_layers: Lista com numero de neuronios por camada oculta
            dropout: Taxa de dropout
            use_residual: Se deve usar conexoes residuais
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256, 128, 128, 64]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.model_name = "DeepMLP"
        
        # Primeira camada
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Camadas ocultas
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                    nn.BatchNorm1d(hidden_layers[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            )
        
        # Camada de saída
        self.output_layer = nn.Linear(hidden_layers[-1], num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.hidden:
            identity = x
            x = layer(x)
            
            # Residual connection (se dimensões compatíveis)
            if self.use_residual and x.shape == identity.shape:
                x = x + identity
        
        x = self.output_layer(x)
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "use_residual": self.use_residual,
            "num_parameters": self.count_parameters()
        }
