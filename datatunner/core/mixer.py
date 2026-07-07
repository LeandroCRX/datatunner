"""
Engine de mistura de dados reais e sintéticos
"""

import numpy as np
import random
from typing import List, Tuple, Union, Dict
from pathlib import Path
import shutil


class DataMixer:
    """Motor de mistura de dados reais e sintéticos"""
    
    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: Seed para reprodutibilidade
        """
        self.random_seed = random_seed
        self.set_seed(random_seed)
    
    def set_seed(self, seed: int):
        """Define seed para reprodutibilidade"""
        random.seed(seed)
        np.random.seed(seed)
    
    def mix_image_data(
        self,
        real_paths: List[str],
        real_labels: List[int],
        synthetic_paths: List[str],
        synthetic_labels: List[int],
        proportion: float,
        balance_classes: bool = True
    ) -> Tuple[List[str], List[int]]:
        """
        Mistura dados de imagem com proporção especificada
        
        Fórmula: D_hybrid = D_real ∪ sample(D_syn, α · |D_real|)
        
        Args:
            real_paths: Caminhos das imagens reais
            real_labels: Labels das imagens reais
            synthetic_paths: Caminhos das imagens sintéticas
            synthetic_labels: Labels das imagens sintéticas
            proportion: Proporção de dados sintéticos a ADICIONAR aos reais.
                       α=0.0 → Apenas dados reais (baseline)
                       α=0.5 → Todos reais + 50% de sintéticos adicionais
                       α=1.0 → Todos reais + 100% de sintéticos adicionais
            balance_classes: Se deve balancear classes
            
        Returns:
            Tupla (mixed_paths, mixed_labels)
        """
        if proportion < 0.0 or proportion > 1.0:
            raise ValueError("Proporção deve estar entre 0.0 e 1.0")
        
        # Se proporção é 0, retorna apenas dados reais (baseline)
        if proportion == 0.0:
            return real_paths.copy(), real_labels.copy()
        
        # Todos os dados reais + proportion * |D_real| sintéticos
        n_real = len(real_paths)
        n_synthetic = int(n_real * proportion)
        
        # Selecionar dados sintéticos
        if balance_classes:
            synthetic_paths_selected, synthetic_labels_selected = self._sample_balanced(
                synthetic_paths, synthetic_labels, n_synthetic
            )
        else:
            indices = random.sample(range(len(synthetic_paths)), min(n_synthetic, len(synthetic_paths)))
            synthetic_paths_selected = [synthetic_paths[i] for i in indices]
            synthetic_labels_selected = [synthetic_labels[i] for i in indices]
        
        # Combinar
        mixed_paths = real_paths + synthetic_paths_selected
        mixed_labels = real_labels + synthetic_labels_selected
        
        # Embaralhar
        combined = list(zip(mixed_paths, mixed_labels))
        random.shuffle(combined)
        mixed_paths, mixed_labels = zip(*combined)
        
        return list(mixed_paths), list(mixed_labels)
    
    def mix_tabular_data(
        self,
        real_data: np.ndarray,
        real_labels: np.ndarray,
        synthetic_data: np.ndarray,
        synthetic_labels: np.ndarray,
        proportion: float,
        balance_classes: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mistura dados tabulares com proporção especificada
        
        Fórmula: D_hybrid = D_real ∪ sample(D_syn, α · |D_real|)
        
        Args:
            real_data: Dados reais (features)
            real_labels: Labels reais
            synthetic_data: Dados sintéticos (features)
            synthetic_labels: Labels sintéticos
            proportion: Proporção de dados sintéticos a ADICIONAR aos reais.
                       α=0.0 → Apenas dados reais (baseline)
                       α=0.5 → Todos reais + 50% de sintéticos adicionais
                       α=1.0 → Todos reais + 100% de sintéticos adicionais
            balance_classes: Se deve balancear classes
            
        Returns:
            Tupla (mixed_data, mixed_labels)
        """
        if proportion < 0.0 or proportion > 1.0:
            raise ValueError("Proporção deve estar entre 0.0 e 1.0")
        
        # Se proporção é 0, retorna apenas dados reais (baseline)
        if proportion == 0.0:
            return real_data.copy(), real_labels.copy()
        
        # Todos os dados reais + proportion * |D_real| sintéticos
        n_real = len(real_data)
        n_synthetic = int(n_real * proportion)
        
        # Selecionar dados sintéticos
        if balance_classes:
            indices = self._get_balanced_indices(synthetic_labels, n_synthetic)
        else:
            indices = random.sample(range(len(synthetic_data)), min(n_synthetic, len(synthetic_data)))
        
        synthetic_data_selected = synthetic_data[indices]
        synthetic_labels_selected = synthetic_labels[indices]
        
        # Combinar
        mixed_data = np.vstack([real_data, synthetic_data_selected])
        mixed_labels = np.hstack([real_labels, synthetic_labels_selected])
        
        # Embaralhar
        indices = np.arange(len(mixed_data))
        np.random.shuffle(indices)
        
        return mixed_data[indices], mixed_labels[indices]
    
    def _sample_balanced(
        self,
        paths: List[str],
        labels: List[int],
        n_samples: int
    ) -> Tuple[List[str], List[int]]:
        """
        Amostragem balanceada por classe
        
        Args:
            paths: Lista de caminhos
            labels: Lista de labels
            n_samples: Número total de amostras desejadas
            
        Returns:
            Tupla (sampled_paths, sampled_labels)
        """
        unique_classes = list(set(labels))
        n_per_class = n_samples // len(unique_classes)
        
        sampled_paths = []
        sampled_labels = []
        
        for class_id in unique_classes:
            class_indices = [i for i, label in enumerate(labels) if label == class_id]
            
            # Se não houver amostras suficientes, pegue todas disponíveis
            n_to_sample = min(n_per_class, len(class_indices))
            selected_indices = random.sample(class_indices, n_to_sample)
            
            sampled_paths.extend([paths[i] for i in selected_indices])
            sampled_labels.extend([labels[i] for i in selected_indices])
        
        return sampled_paths, sampled_labels
    
    def _get_balanced_indices(
        self,
        labels: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Retorna índices balanceados por classe
        
        Args:
            labels: Array de labels
            n_samples: Número total de amostras
            
        Returns:
            Array de índices
        """
        if n_samples <= 0 or len(labels) == 0:
            return np.array([], dtype=int)

        unique_classes = np.unique(labels)
        if len(unique_classes) == 0:
            return np.array([], dtype=int)

        n_per_class = n_samples // len(unique_classes)
        
        indices = []
        
        for class_id in unique_classes:
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) == 0:
                continue
            
            n_to_sample = min(n_per_class, len(class_indices))
            selected_indices = np.random.choice(class_indices, n_to_sample, replace=False)
            
            indices.extend(selected_indices.tolist())
        
        if not indices:
            return np.array([], dtype=int)
        
        return np.array(indices)
    
    def get_mixture_stats(
        self,
        real_labels: Union[List[int], np.ndarray],
        synthetic_labels: Union[List[int], np.ndarray],
        proportion: float
    ) -> Dict[str, Union[int, float]]:
        """
        Calcula estatísticas da mistura
        
        Args:
            real_labels: Labels dos dados reais
            synthetic_labels: Labels dos dados sintéticos
            proportion: Proporção de sintéticos (α)
            
        Returns:
            Dicionário com estatísticas
        """
        n_real = len(real_labels)
        n_synthetic = int(n_real * proportion)
        n_total = n_real + n_synthetic
        
        stats = {
            "n_real": n_real,
            "n_synthetic": n_synthetic,
            "n_total": n_total,
            "proportion_added": proportion,
            "effective_proportion": n_synthetic / n_total if n_total > 0 else 0,
        }
        
        return stats
