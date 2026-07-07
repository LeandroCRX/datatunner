"""
Carregamento e pré-processamento de dados
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Optional, Union, Dict
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ImageDataset(Dataset):
    """Dataset customizado para imagens"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            image_paths: Lista de caminhos para imagens
            labels: Lista de labels
            transform: Transformações a serem aplicadas
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TabularDataset(Dataset):
    """Dataset customizado para dados tabulares"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ):
        """
        Args:
            features: Features (X)
            labels: Labels (y)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class DataLoader:
    """Carregador de dados para imagens e dados tabulares"""
    
    @staticmethod
    def load_image_data(
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ) -> Tuple[List[str], List[int], List[str]]:
        """
        Carrega dados de imagem de uma estrutura de diretórios
        
        Estrutura esperada:
        data_dir/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg
                img4.jpg
        
        Args:
            data_dir: Diretório raiz com subpastas de classes
            image_size: Tamanho para redimensionar imagens
            normalize: Se deve normalizar as imagens
            
        Returns:
            Tupla (image_paths, labels, class_names)
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise ValueError(f"Diretório não encontrado: {data_dir}")
        
        # Detectar classes automaticamente
        class_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
        class_names = [d.name for d in class_folders]
        
        print(f"Classes detectadas: {class_names}")
        
        image_paths = []
        labels = []
        
        for class_idx, class_folder in enumerate(class_folders):
            # Extensões de imagem suportadas
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            
            for ext in extensions:
                for img_path in class_folder.glob(ext):
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
        
        print(f"Total de imagens carregadas: {len(image_paths)}")
        print(f"Distribuição por classe: {np.bincount(labels)}")
        
        return image_paths, labels, class_names
    
    @staticmethod
    def get_image_transforms(
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        normalize: bool = True
    ) -> transforms.Compose:
        """
        Retorna transformações para imagens
        
        Args:
            image_size: Tamanho da imagem
            augment: Se deve aplicar data augmentation
            normalize: Se deve normalizar
            
        Returns:
            Composição de transformações
        """
        transform_list = []
        
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        
        transform_list.append(transforms.Resize(image_size))
        transform_list.append(transforms.ToTensor())
        
        if normalize:
            # ImageNet statistics
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def load_tabular_data(
        file_path: str,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Carrega dados tabulares de um arquivo CSV
        
        Args:
            file_path: Caminho para o arquivo CSV
            target_column: Nome da coluna target
            test_size: Proporção do conjunto de teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tupla (X_train, y_train, X_test, y_test)
        """
        df = pd.read_csv(file_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Dados carregados: {len(df)} amostras")
        print(f"Features: {X.shape[1]}")
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def preprocess_tabular_data(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Pré-processa dados tabulares
        
        Args:
            X_train: Features de treino
            X_test: Features de teste
            categorical_columns: Colunas categóricas
            numerical_columns: Colunas numéricas
            
        Returns:
            Tupla (X_train_processed, X_test_processed, preprocessing_info)
        """
        preprocessing_info = {}
        
        # Detectar automaticamente se não especificado
        if categorical_columns is None:
            categorical_columns = X_train.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        
        if numerical_columns is None:
            numerical_columns = X_train.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
        
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Encoding de variáveis categóricas
        if categorical_columns:
            X_train_processed = pd.get_dummies(
                X_train_processed, columns=categorical_columns, drop_first=True
            )
            X_test_processed = pd.get_dummies(
                X_test_processed, columns=categorical_columns, drop_first=True
            )
            
            # Garantir mesmas colunas
            missing_cols = set(X_train_processed.columns) - set(X_test_processed.columns)
            for col in missing_cols:
                X_test_processed[col] = 0
            
            X_test_processed = X_test_processed[X_train_processed.columns]
        
        # Normalização de variáveis numéricas
        if numerical_columns:
            scaler = StandardScaler()
            
            numerical_cols_processed = [
                col for col in numerical_columns if col in X_train_processed.columns
            ]
            
            X_train_processed[numerical_cols_processed] = scaler.fit_transform(
                X_train_processed[numerical_cols_processed]
            )
            X_test_processed[numerical_cols_processed] = scaler.transform(
                X_test_processed[numerical_cols_processed]
            )
            
            preprocessing_info['scaler'] = scaler
        
        preprocessing_info['categorical_columns'] = categorical_columns
        preprocessing_info['numerical_columns'] = numerical_columns
        preprocessing_info['feature_names'] = X_train_processed.columns.tolist()
        
        return (
            X_train_processed.values,
            X_test_processed.values,
            preprocessing_info
        )


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    validation_split: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Cria data loaders de treino e validação
    
    Args:
        dataset: Dataset PyTorch
        batch_size: Tamanho do batch
        shuffle: Se deve embaralhar
        num_workers: Número de workers
        validation_split: Proporção de validação
        random_state: Seed
        
    Returns:
        Tupla (train_loader, val_loader)
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers
    )
    
    return train_loader, val_loader
