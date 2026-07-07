"""
Engine de mistura de dados reais e sinteticos
"""

import random
import numpy as np
from typing import List, Tuple, Union, Dict


class DataMixer:
    """Motor de mistura de dados reais e sinteticos"""

    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: Seed para reprodutibilidade
        """
        self.random_seed = random_seed
        self.py_rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def set_seed(self, seed: int):
        """Define seed para reprodutibilidade"""
        self.random_seed = seed
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
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
        Mistura dados de imagem com proporcao especificada

        Formula: D_hybrid = D_real U sample(D_syn, a * |D_real|)

        Args:
            real_paths: Caminhos das imagens reais
            real_labels: Labels das imagens reais
            synthetic_paths: Caminhos das imagens sinteticas
            synthetic_labels: Labels das imagens sinteticas
            proportion: Proporcao de dados sinteticos a ADICIONAR aos reais.
                        a=0.0 -> Apenas dados reais (baseline)
                        a=0.5 -> Todos reais + 50% de sinteticos adicionais
                        a=1.0 -> Todos reais + 100% de sinteticos adicionais
            balance_classes: Se deve balancear classes

        Returns:
            Tupla (mixed_paths, mixed_labels)
        """
        if len(real_paths) != len(real_labels):
            raise ValueError("real_paths e real_labels devem ter o mesmo tamanho")
        if len(synthetic_paths) != len(synthetic_labels):
            raise ValueError("synthetic_paths e synthetic_labels devem ter o mesmo tamanho")
        if proportion < 0.0 or proportion > 1.0:
            raise ValueError("Proporcao deve estar entre 0.0 e 1.0")

        if proportion == 0.0:
            combined = list(zip(real_paths.copy(), real_labels.copy()))
            self.py_rng.shuffle(combined)
            if not combined:
                return [], []
            mixed_paths, mixed_labels = zip(*combined)
            return list(mixed_paths), list(mixed_labels)

        if len(synthetic_paths) == 0:
            raise ValueError("synthetic_paths vazio mas proportion > 0")

        n_real = len(real_paths)
        n_synthetic = int(n_real * proportion)

        if balance_classes:
            synthetic_paths_selected, synthetic_labels_selected = self._sample_balanced(
                synthetic_paths, synthetic_labels, n_synthetic
            )
        else:
            indices = self.py_rng.sample(
                range(len(synthetic_paths)),
                min(n_synthetic, len(synthetic_paths))
            )
            synthetic_paths_selected = [synthetic_paths[i] for i in indices]
            synthetic_labels_selected = [synthetic_labels[i] for i in indices]

        mixed_paths = list(real_paths) + synthetic_paths_selected
        mixed_labels = list(real_labels) + synthetic_labels_selected

        combined = list(zip(mixed_paths, mixed_labels))
        self.py_rng.shuffle(combined)
        if not combined:
            return [], []
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
        Mistura dados tabulares com proporcao especificada

        Formula: D_hybrid = D_real U sample(D_syn, a * |D_real|)

        Args:
            real_data: Dados reais (features)
            real_labels: Labels reais
            synthetic_data: Dados sinteticos (features)
            synthetic_labels: Labels sinteticos
            proportion: Proporcao de dados sinteticos a ADICIONAR aos reais.
            balance_classes: Se deve balancear classes

        Returns:
            Tupla (mixed_data, mixed_labels)
        """
        if len(real_data) != len(real_labels):
            raise ValueError("real_data e real_labels devem ter o mesmo tamanho")
        if len(synthetic_data) != len(synthetic_labels):
            raise ValueError("synthetic_data e synthetic_labels devem ter o mesmo tamanho")
        if proportion < 0.0 or proportion > 1.0:
            raise ValueError("Proporcao deve estar entre 0.0 e 1.0")

        if proportion == 0.0:
            indices = self.np_rng.permutation(len(real_data))
            return real_data[indices].copy(), real_labels[indices].copy()

        if len(synthetic_data) == 0:
            raise ValueError("synthetic_data vazio mas proportion > 0")

        n_real = len(real_data)
        n_synthetic = int(n_real * proportion)

        if balance_classes:
            indices = self._get_balanced_indices(synthetic_labels, n_synthetic)
        else:
            indices = self.np_rng.choice(
                len(synthetic_data),
                min(n_synthetic, len(synthetic_data)),
                replace=False
            )

        if len(indices) == 0:
            indices = self.np_rng.choice(
                len(synthetic_data),
                min(n_synthetic, len(synthetic_data)),
                replace=False
            )

        synthetic_data_selected = synthetic_data[indices]
        synthetic_labels_selected = synthetic_labels[indices]

        mixed_data = np.vstack([real_data, synthetic_data_selected])
        mixed_labels = np.hstack([real_labels, synthetic_labels_selected])

        perm = self.np_rng.permutation(len(mixed_data))

        return mixed_data[perm], mixed_labels[perm]

    def _sample_balanced(
        self,
        paths: List[str],
        labels: List[int],
        n_samples: int
    ) -> Tuple[List[str], List[int]]:
        """Amostragem balanceada por classe"""
        if n_samples <= 0 or len(labels) == 0:
            return [], []

        unique_classes = sorted(set(labels))
        n_per_class, remainder = divmod(n_samples, len(unique_classes))

        sampled_paths = []
        sampled_labels = []

        for i, class_id in enumerate(unique_classes):
            class_indices = [j for j, label in enumerate(labels) if label == class_id]
            if not class_indices:
                continue

            n_to_sample = n_per_class + (1 if i < remainder else 0)
            n_to_sample = min(n_to_sample, len(class_indices))
            if n_to_sample <= 0:
                continue

            selected_indices = self.py_rng.sample(class_indices, n_to_sample)
            sampled_paths.extend([paths[j] for j in selected_indices])
            sampled_labels.extend([labels[j] for j in selected_indices])

        return sampled_paths, sampled_labels

    def _get_balanced_indices(
        self,
        labels: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Retorna indices balanceados por classe"""
        if n_samples <= 0 or len(labels) == 0:
            return np.array([], dtype=int)

        unique_classes = np.unique(labels)
        if len(unique_classes) == 0:
            return np.array([], dtype=int)

        n_per_class, remainder = divmod(n_samples, len(unique_classes))

        indices = []

        for i, class_id in enumerate(unique_classes):
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) == 0:
                continue

            n_to_sample = n_per_class + (1 if i < remainder else 0)
            n_to_sample = min(n_to_sample, len(class_indices))
            if n_to_sample <= 0:
                continue

            selected_indices = self.np_rng.choice(class_indices, n_to_sample, replace=False)
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
        """Calcula estatisticas da mistura"""
        n_real = len(real_labels)
        n_synthetic = int(n_real * proportion)
        n_total = n_real + n_synthetic

        stats = {
            "n_real": n_real,
            "n_synthetic": n_synthetic,
            "n_total": n_total,
            "proportion_added": proportion,
            "effective_proportion": n_synthetic / n_total if n_total > 0 else 0.0,
        }

        return stats
