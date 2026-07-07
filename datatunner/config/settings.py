"""
Configurações padrão do DataTunner
"""

import os
from typing import Dict, Any

# Diretórios padrão
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_LOG_DIR = "logs"

# Configurações de reprodutibilidade
DEFAULT_RANDOM_SEED = 42
DEFAULT_DETERMINISTIC = True

# Configurações de treinamento
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_OPTIMIZER = "adam"

# Proporções padrão para teste
DEFAULT_PROPORTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Configurações de validação
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.2

# Configurações de dispositivo
DEFAULT_DEVICE = "cuda"  # cuda, cpu, mps
DEFAULT_NUM_WORKERS = 4

# Métricas padrão
CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]

REGRESSION_METRICS = [
    "mse",
    "rmse",
    "mae",
    "r2_score",
]

# Configurações de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configurações de visualização
PLOT_STYLE = "auto"
PLOT_DPI = 150
PLOT_FIGSIZE = (12, 8)

# Configurações de checkpointing
CHECKPOINT_FREQUENCY = 5  # Salvar a cada N épocas
SAVE_BEST_ONLY = True

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001


def get_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Retorna configuração completa mesclando defaults com configurações customizadas
    
    Args:
        custom_config: Dicionário com configurações customizadas
        
    Returns:
        Configuração completa
    """
    config = {
        "output_dir": DEFAULT_OUTPUT_DIR,
        "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
        "log_dir": DEFAULT_LOG_DIR,
        "random_seed": DEFAULT_RANDOM_SEED,
        "deterministic": DEFAULT_DETERMINISTIC,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "optimizer": DEFAULT_OPTIMIZER,
        "proportions": DEFAULT_PROPORTIONS,
        "validation_split": DEFAULT_VALIDATION_SPLIT,
        "test_split": DEFAULT_TEST_SPLIT,
        "device": DEFAULT_DEVICE,
        "num_workers": DEFAULT_NUM_WORKERS,
        "classification_metrics": CLASSIFICATION_METRICS,
        "regression_metrics": REGRESSION_METRICS,
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        "plot_style": PLOT_STYLE,
        "plot_dpi": PLOT_DPI,
        "plot_figsize": PLOT_FIGSIZE,
        "checkpoint_frequency": CHECKPOINT_FREQUENCY,
        "save_best_only": SAVE_BEST_ONLY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    }
    
    if custom_config:
        config.update(custom_config)
    
    return config


def setup_directories(base_dir: str = "."):
    """
    Cria estrutura de diretórios necessária
    
    Args:
        base_dir: Diretório base
    """
    dirs = [
        DEFAULT_OUTPUT_DIR,
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_LOG_DIR,
    ]
    
    for dir_name in dirs:
        path = os.path.join(base_dir, dir_name)
        os.makedirs(path, exist_ok=True)
