"""
CTGAN para geração de dados tabulares sintéticos
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, List
from pathlib import Path
import warnings

try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False
    warnings.warn(
        "SDV não encontrado. Instale com: pip install sdv>=1.2.0",
        ImportWarning
    )

from datatunner.generators.base import BaseSyntheticGenerator


class CTGANGenerator(BaseSyntheticGenerator):
    """
    Gerador de dados tabulares usando CTGAN (Conditional Tabular GAN)
    
    CTGAN é uma GAN especialmente desenvolvida para dados tabulares,
    capaz de lidar com variáveis mistas (contínuas e categóricas).
    """
    
    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: Tuple[int, ...] = (256, 256),
        discriminator_dim: Tuple[int, ...] = (256, 256),
        embedding_dim: int = 128,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        discriminator_steps: int = 1,
        random_seed: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            epochs: Número de épocas de treinamento
            batch_size: Tamanho do batch
            generator_dim: Dimensões das camadas do gerador
            discriminator_dim: Dimensões das camadas do discriminador
            embedding_dim: Dimensão do embedding
            generator_lr: Taxa de aprendizado do gerador
            discriminator_lr: Taxa de aprendizado do discriminador
            discriminator_steps: Passos do discriminador por passo do gerador
            random_seed: Seed para reprodutibilidade
            verbose: Se deve exibir progresso
        """
        super().__init__(random_seed)
        
        if not CTGAN_AVAILABLE:
            raise ImportError(
                "SDV não está instalado. "
                "Instale com: pip install sdv>=1.2.0"
            )
        
        self.generator_name = "CTGAN"
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.embedding_dim = embedding_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.verbose = verbose
        
        self.model = None
        self.metadata = None
        self.column_names = None
        self.categorical_columns = None
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Optional[np.ndarray] = None,
        categorical_columns: Optional[List[str]] = None
    ):
        """
        Treina CTGAN nos dados
        
        Args:
            data: Dados de treino (DataFrame ou array)
            labels: Labels (opcional, serão incluídos nos dados)
            categorical_columns: Lista de colunas categóricas
        """
        # Converter para DataFrame se necessário
        if isinstance(data, np.ndarray):
            self.column_names = [f"feature_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=self.column_names)
        else:
            df = data.copy()
            self.column_names = list(df.columns)
        
        # Adicionar labels se fornecidos
        if labels is not None:
            df['target'] = labels
            if categorical_columns is None:
                categorical_columns = []
            categorical_columns = list(categorical_columns) + ['target']
        
        self.categorical_columns = categorical_columns or []
        
        # Criar metadados automaticamente
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(df)
        
        # Atualizar tipos de colunas categóricas se especificadas
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    self.metadata.update_column(col, sdtype='categorical')
        
        # Criar e configurar modelo CTGAN
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            embedding_dim=self.embedding_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
            verbose=self.verbose,
            cuda=False  # Será definido automaticamente pela SDV
        )
        
        print(f"Treinando CTGAN por {self.epochs} épocas...")
        self.model.fit(df)
        print("✅ CTGAN treinado com sucesso!")
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict] = None
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        Gera dados sintéticos usando CTGAN
        
        Args:
            n_samples: Número de amostras a gerar
            conditions: Condições para geração condicional (opcional)
                       Exemplo: {'target': 1} para gerar apenas classe 1
            
        Returns:
            DataFrame com dados sintéticos ou tupla (X, y) se houver target
        """
        if self.model is None:
            raise ValueError("Execute fit() antes de generate()")
        
        print(f"Gerando {n_samples} amostras sintéticas com CTGAN...")
        
        if conditions:
            # Geração condicional
            synthetic_data = self.model.sample_from_conditions(
                conditions=[conditions] * n_samples
            )
        else:
            # Geração normal
            synthetic_data = self.model.sample(num_rows=n_samples)
        
        # Se houver coluna target, separar features e labels
        if 'target' in synthetic_data.columns:
            y_synthetic = synthetic_data['target'].values
            X_synthetic = synthetic_data.drop(columns=['target'])
            
            # Converter para arrays numpy se os dados originais eram arrays
            if not isinstance(self.column_names, list) or \
               all(col.startswith('feature_') for col in self.column_names):
                return X_synthetic.values, y_synthetic
            else:
                return X_synthetic, y_synthetic
        
        return synthetic_data
    
    def generate_balanced(
        self,
        n_samples_per_class: int,
        target_column: str = 'target'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Gera dados balanceados para cada classe
        
        Args:
            n_samples_per_class: Número de amostras por classe
            target_column: Nome da coluna target
            
        Returns:
            Tupla (features, labels) balanceada
        """
        if self.model is None:
            raise ValueError("Execute fit() antes de generate_balanced()")
        
        # Obter classes únicas do metadata
        target_values = self.metadata.columns[target_column].get('values', None)
        
        if target_values is None:
            # Tentar inferir das condições possíveis
            raise ValueError(
                f"Não foi possível determinar valores únicos para {target_column}"
            )
        
        all_samples = []
        all_labels = []
        
        for class_value in target_values:
            print(f"Gerando {n_samples_per_class} amostras para classe {class_value}...")
            
            synthetic = self.model.sample_from_conditions(
                conditions=[{target_column: class_value}] * n_samples_per_class
            )
            
            y = synthetic[target_column].values
            X = synthetic.drop(columns=[target_column])
            
            all_samples.append(X)
            all_labels.append(y)
        
        # Concatenar e embaralhar
        X_synthetic = pd.concat(all_samples, ignore_index=True)
        y_synthetic = np.concatenate(all_labels)
        
        # Embaralhar
        indices = np.random.permutation(len(X_synthetic))
        X_synthetic = X_synthetic.iloc[indices].reset_index(drop=True)
        y_synthetic = y_synthetic[indices]
        
        return X_synthetic, y_synthetic
    
    def save_model(self, filepath: str):
        """
        Salva modelo treinado
        
        Args:
            filepath: Caminho para salvar
        """
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar")
        
        self.model.save(filepath)
        print(f"✅ Modelo CTGAN salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carrega modelo salvo
        
        Args:
            filepath: Caminho do modelo
        """
        self.model = CTGANSynthesizer.load(filepath)
        print(f"✅ Modelo CTGAN carregado de: {filepath}")
    
    def get_generator_info(self) -> Dict:
        """Retorna informações sobre o gerador"""
        info = super().get_generator_info()
        info.update({
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'generator_dim': self.generator_dim,
            'discriminator_dim': self.discriminator_dim,
            'embedding_dim': self.embedding_dim,
            'generator_lr': self.generator_lr,
            'discriminator_lr': self.discriminator_lr,
            'categorical_columns': self.categorical_columns,
            'is_trained': self.model is not None
        })
        return info


class TVAEGenerator(BaseSyntheticGenerator):
    """
    Gerador usando TVAE (Tabular Variational AutoEncoder)
    
    Alternativa ao CTGAN, geralmente mais rápido e melhor para
    datasets menores.
    """
    
    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        compress_dims: Tuple[int, ...] = (128, 128),
        decompress_dims: Tuple[int, ...] = (128, 128),
        random_seed: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            epochs: Número de épocas
            batch_size: Tamanho do batch
            embedding_dim: Dimensão do embedding
            compress_dims: Dimensões do encoder
            decompress_dims: Dimensões do decoder
            random_seed: Seed
            verbose: Verbosidade
        """
        super().__init__(random_seed)
        
        if not CTGAN_AVAILABLE:
            raise ImportError("SDV não está instalado")
        
        try:
            from sdv.single_table import TVAESynthesizer
            self.TVAESynthesizer = TVAESynthesizer
        except ImportError:
            raise ImportError("TVAE não disponível na sua versão do SDV")
        
        self.generator_name = "TVAE"
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.verbose = verbose
        
        self.model = None
        self.metadata = None
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Optional[np.ndarray] = None,
        categorical_columns: Optional[List[str]] = None
    ):
        """Treina TVAE nos dados"""
        # Converter para DataFrame
        if isinstance(data, np.ndarray):
            column_names = [f"feature_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=column_names)
        else:
            df = data.copy()
        
        # Adicionar labels
        if labels is not None:
            df['target'] = labels
        
        # Criar metadados
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(df)
        
        # Criar modelo
        self.model = self.TVAESynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            verbose=self.verbose
        )
        
        print(f"Treinando TVAE por {self.epochs} épocas...")
        self.model.fit(df)
        print("✅ TVAE treinado!")
    
    def generate(self, n_samples: int) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Gera dados sintéticos"""
        if self.model is None:
            raise ValueError("Execute fit() primeiro")
        
        synthetic_data = self.model.sample(num_rows=n_samples)
        
        if 'target' in synthetic_data.columns:
            y = synthetic_data['target'].values
            X = synthetic_data.drop(columns=['target'])
            return X.values, y
        
        return synthetic_data
    
    def get_generator_info(self) -> Dict:
        info = super().get_generator_info()
        info.update({
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'is_trained': self.model is not None
        })
        return info
