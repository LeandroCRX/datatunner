"""
Data Augmentation para imagens
"""

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional
import os
from pathlib import Path

from datatunner.generators.base import BaseSyntheticGenerator


class ImageAugmentation(BaseSyntheticGenerator):
    """Gerador de imagens sintéticas via Data Augmentation"""
    
    def __init__(
        self,
        random_seed: int = 42,
        augmentation_strength: str = "medium"
    ):
        """
        Args:
            random_seed: Seed para reprodutibilidade
            augmentation_strength: Intensidade da augmentation (light, medium, heavy)
        """
        super().__init__(random_seed)
        
        self.generator_name = "ImageAugmentation"
        self.augmentation_strength = augmentation_strength
        self.transform = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Cria pipeline de augmentation"""
        
        if self.augmentation_strength == "light":
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=15, p=0.3),
            ])
        
        elif self.augmentation_strength == "medium":
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Rotate(limit=30, p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5
                ),
                A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
            ])
        
        elif self.augmentation_strength == "heavy":
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.Rotate(limit=45, p=0.6),
                A.Affine(
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    scale=(0.8, 1.2),
                    rotate=(-30, 30),
                    p=0.6
                ),
                A.GaussNoise(std_range=(0.02, 0.20), p=0.4),
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
                A.CoarseDropout(
                    num_holes_range=(4, 8),
                    hole_height_range=(16, 32),
                    hole_width_range=(16, 32),
                    p=0.3
                ),
            ])
        
        else:
            raise ValueError(f"Intensidade inválida: {self.augmentation_strength}")
        
        return transform
    
    def fit(self, data: List[str], labels: Optional[np.ndarray] = None):
        """
        Para augmentation, não precisa fit
        
        Args:
            data: Lista de caminhos de imagens
            labels: Labels (não usado)
        """
        self.image_paths = data
        self.labels = labels
    
    def generate(
        self,
        n_samples: int,
        output_dir: Optional[str] = None,
        save_images: bool = False
    ) -> Tuple[List[str], List[int]]:
        """
        Gera imagens sintéticas via augmentation
        
        Args:
            n_samples: Número de imagens a gerar
            output_dir: Diretório para salvar imagens (se save_images=True)
            save_images: Se deve salvar imagens em disco
            
        Returns:
            Tupla (image_paths, labels)
        """
        if not hasattr(self, 'image_paths'):
            raise ValueError("Execute fit() antes de generate()")
        
        augmented_paths = []
        augmented_labels = []
        
        for i in range(n_samples):
            # Selecionar imagem aleatória
            idx = np.random.randint(len(self.image_paths))
            img_path = self.image_paths[idx]
            label = self.labels[idx] if self.labels is not None else 0
            
            # Carregar e aplicar augmentation
            image = np.array(Image.open(img_path).convert('RGB'))
            augmented = self.transform(image=image)
            augmented_image = augmented['image']
            
            if save_images and output_dir:
                # Salvar imagem augmentada
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                aug_img_path = output_path / f"aug_{i}_{Path(img_path).name}"
                Image.fromarray(augmented_image).save(aug_img_path)
                
                augmented_paths.append(str(aug_img_path))
            else:
                # Se não salvar, manter referência original (será processado em runtime)
                augmented_paths.append(img_path)
            
            augmented_labels.append(label)
        
        return augmented_paths, augmented_labels
    
    def generate_to_directory(
        self,
        input_dir: str,
        output_dir: str,
        samples_per_image: int = 1,
        max_samples_per_class: Optional[int] = None
    ) -> None:
        """
        Lê imagens de um diretório (com subpastas de classes) e salva versões augmentadas
        
        Args:
            input_dir: Diretório com imagens originais
            output_dir: Diretório de destino
            samples_per_image: Quantas versões gerar por imagem original
            max_samples_per_class: Limite de imagens por classe (opcional)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Detectar classes (subpastas)
        classes = [d.name for d in input_path.iterdir() if d.is_dir()]
        
        if not classes:
            # Tentar processar como diretório plano se não houver subpastas
            classes = ["."]
            
        print(f"🚀 Iniciando augmentation de {len(classes)} classes...")
        
        for class_name in classes:
            class_in = input_path / class_name
            class_out = output_path / class_name
            class_out.mkdir(parents=True, exist_ok=True)
            
            # Listar imagens (extensões comuns)
            img_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                img_files.extend(list(class_in.glob(ext)))
            
            if not img_files:
                continue
                
            print(f"   Processando '{class_name}': {len(img_files)} originais...")
            
            count = 0
            # Embaralhar para pegar aleatórios se houver limite
            np.random.shuffle(img_files)
            
            for img_file in img_files:
                if max_samples_per_class and count >= max_samples_per_class:
                    break
                    
                image = np.array(Image.open(img_file).convert('RGB'))
                
                for s in range(samples_per_image):
                    if max_samples_per_class and count >= max_samples_per_class:
                        break
                        
                    # Aplicar augmentation
                    augmented = self.transform(image=image)['image']
                    
                    # Salvar
                    save_path = class_out / f"aug_{count}_{img_file.name}"
                    Image.fromarray(augmented).save(save_path)
                    count += 1
                    
        print(f"✅ Augmentation concluída! Resultados em {output_dir}")

    def get_generator_info(self):
        info = super().get_generator_info()
        info['augmentation_strength'] = self.augmentation_strength
        return info
