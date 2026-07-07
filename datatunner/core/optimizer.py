"""
Motor principal de otimização do DataTunner
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime
import logging

from datatunner.core.mixer import DataMixer
from datatunner.core.evaluator import ModelEvaluator
from datatunner.utils.data_loader import (
    DataLoader as DTDataLoader,
    ImageDataset,
    TabularDataset,
    create_data_loaders
)
from datatunner.utils.visualization import ResultsVisualizer
from datatunner.config.settings import get_config, setup_directories


class DataTunner:
    """
    Motor principal de otimização de proporções de dados sintéticos
    """
    
    def __init__(
        self,
        data_type: str = "image",
        real_data_path: Optional[str] = None,
        synthetic_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        output_dir: str = "results",
        random_seed: int = 42,
        config: Optional[Dict] = None
    ):
        """
        Args:
            data_type: Tipo de dados ('image' ou 'tabular')
            real_data_path: Caminho para dados reais
            synthetic_data_path: Caminho para dados sintéticos
            test_data_path: Caminho para dados de teste (opcional)
            output_dir: Diretório de saída
            random_seed: Seed para reprodutibilidade
            config: Configurações customizadas
        """
        self.data_type = data_type.lower()
        self.real_data_path = real_data_path
        self.synthetic_data_path = synthetic_data_path
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Configuração
        self.config = get_config(config)
        
        # Criar diretórios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_directories(str(self.output_dir))
        
        # Componentes
        self.mixer = DataMixer(random_seed=random_seed)
        self.evaluator = None
        self.visualizer = ResultsVisualizer(
            output_dir=str(self.output_dir),
            style=self.config.get('plot_style', 'auto')
        )
        
        # Dados
        self.real_data = None
        self.synthetic_data = None
        self.test_data = None
        self.class_names = None
        
        # Resultados
        self.results = {}
        self.best_proportion = None
        self.best_metrics = None
        
        # Reprodutibilidade
        self._set_seeds()
        
        # Logging
        self._setup_logging()
        
        # Carregar dados
        if real_data_path:
            self._load_data()
    
    def _set_seeds(self):
        """Configura seeds para reprodutibilidade"""
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_logging(self):
        """Configura sistema de logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"datatunner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format=self.config['log_format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataTunner inicializado")
        self.logger.info(f"Tipo de dados: {self.data_type}")
        self.logger.info(f"Random seed: {self.random_seed}")
    
    def _load_data(self):
        """Carrega dados baseado no tipo"""
        self.logger.info("Carregando dados...")
        
        if self.data_type == "image":
            self._load_image_data()
        elif self.data_type == "tabular":
            self._load_tabular_data()
        else:
            raise ValueError(f"Tipo de dados não suportado: {self.data_type}")
    
    def _load_image_data(self):
        """Carrega dados de imagem"""
        loader = DTDataLoader()
        
        # Dados reais
        self.real_paths, self.real_labels, self.class_names = loader.load_image_data(
            self.real_data_path
        )
        
        # Dados sintéticos
        if self.synthetic_data_path:
            self.synthetic_paths, self.synthetic_labels, _ = loader.load_image_data(
                self.synthetic_data_path
            )
        
        # Dados de teste (se fornecidos)
        if self.test_data_path:
            self.test_paths, self.test_labels, _ = loader.load_image_data(
                self.test_data_path
            )
        
        self.logger.info(f"Classes detectadas: {self.class_names}")
        self.logger.info(f"Dados reais: {len(self.real_paths)} imagens")
        if hasattr(self, 'synthetic_paths'):
            self.logger.info(f"Dados sintéticos: {len(self.synthetic_paths)} imagens")
    
    def _load_tabular_data(self):
        """Carrega dados tabulares de arquivos CSV"""
        import pandas as pd

        loader = DTDataLoader()

        self.logger.info(f"Carregando dados tabulares de: {self.real_data_path}")

        X_train, y_train, X_test, y_test = loader.load_tabular_data(
            self.real_data_path,
            target_column='target',
            test_size=self.config['test_split'],
            random_state=self.random_seed
        )

        self.real_data = X_train.values if hasattr(X_train, 'values') else X_train
        self.real_labels = y_train.values if hasattr(y_train, 'values') else y_train
        self.test_data = X_test.values if hasattr(X_test, 'values') else X_test
        self.test_labels = y_test.values if hasattr(y_test, 'values') else y_test

        if self.synthetic_data_path:
            df_syn = pd.read_csv(self.synthetic_data_path)
            if 'target' in df_syn.columns:
                self.synthetic_data = df_syn.drop(columns=['target']).values
                self.synthetic_labels = df_syn['target'].values
            else:
                self.synthetic_data = df_syn.values
                self.synthetic_labels = None
            self.logger.info(f"Dados sintéticos: {len(self.synthetic_data)} amostras")

        self.class_names = [str(c) for c in np.unique(self.real_labels)]
        self.logger.info(f"Classes detectadas: {self.class_names}")
        self.logger.info(f"Dados reais: {len(self.real_data)} amostras")
    
    def optimize(
        self,
        model: nn.Module = None,
        model_factory=None,
        proportions: List[float] = None,
        synthetic_data: Optional[Any] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        n_trials: int = 1,
        balance_classes: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executa otimização para encontrar melhor proporção
        
        Args:
            model: Instância do modelo PyTorch (usada como template)
            model_factory: Função callable que retorna nova instância do modelo
                          a cada chamada. Se fornecida, é usada em vez de model.
            proportions: Lista de proporções a testar
            synthetic_data: Dados sintéticos (para dados tabulares)
            epochs: Número de épocas
            batch_size: Tamanho do batch
            learning_rate: Taxa de aprendizado
            n_trials: Número de repetições por proporção
            balance_classes: Se deve balancear classes
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com resultados
        """
        if model is None and model_factory is None:
            raise ValueError("Forneça 'model' ou 'model_factory'")

        if model is not None:
            self._model_name = type(model).__name__
        elif model_factory is not None:
            try:
                self._model_name = type(model_factory()).__name__
            except Exception:
                self._model_name = "CustomModel"
        if proportions is None:
            proportions = self.config['proportions']
        
        self.logger.info(f"Iniciando otimização com proporções: {proportions}")
        self.logger.info(f"Épocas: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Configurar evaluator
        task_type = "classification"  # Pode ser parametrizado
        device = self.config['device']
        self.evaluator = ModelEvaluator(device=device, task_type=task_type)
        
        results = {}
        
        for proportion in proportions:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testando proporção: {proportion:.1%}")
            self.logger.info(f"{'='*60}")
            
            trial_results = []
            
            for trial in range(n_trials):
                self.logger.info(f"\nTrial {trial + 1}/{n_trials}")
                
                # Criar dataset misto
                mixed_dataset = self._create_mixed_dataset(
                    proportion, balance_classes
                )
                
                # Criar data loaders
                train_loader, val_loader = create_data_loaders(
                    mixed_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=self.config['num_workers'],
                    validation_split=self.config['validation_split'],
                    random_state=self.random_seed + trial
                )
                
                # Criar test loader
                test_loader = self._create_test_loader(batch_size)
                
                # Reinicializar modelo
                if model_factory is not None:
                    model_instance = model_factory()
                else:
                    model_instance = self._reset_model(model)
                
                # Configurar otimizador e loss
                optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                
                # Treinar e avaliar
                checkpoint_path = str(
                    self.output_dir / "checkpoints" / f"model_prop_{proportion:.2f}_trial_{trial}.pth"
                )
                
                test_metrics, history = self.evaluator.train_and_evaluate(
                    model_instance,
                    train_loader,
                    val_loader,
                    test_loader,
                    optimizer,
                    criterion,
                    epochs=epochs,
                    early_stopping_patience=self.config['early_stopping_patience'],
                    checkpoint_path=checkpoint_path,
                    class_names=self.class_names
                )
                
                trial_results.append(test_metrics)
                
                self.logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
                self.logger.info(f"F1-Score: {test_metrics['f1_score']:.4f}")
            
            # Agregar resultados dos trials
            aggregated_metrics = self._aggregate_trial_results(trial_results)
            results[proportion] = aggregated_metrics
            
            self.logger.info(f"\nMédia - Accuracy: {aggregated_metrics['accuracy']:.4f}")
        
        self.results = results
        
        # Encontrar melhor proporção
        self._find_best_proportion()
        
        # Salvar resultados
        self._save_results()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"MELHOR PROPORÇÃO: {self.best_proportion:.1%}")
        self.logger.info(f"MELHOR ACCURACY: {self.best_metrics['accuracy']:.4f}")
        self.logger.info(f"{'='*60}\n")
        
        return {
            "results": self.results,
            "best_proportion": self.best_proportion,
            "best_metrics": self.best_metrics
        }
    
    def _create_mixed_dataset(
        self,
        proportion: float,
        balance_classes: bool
    ) -> Union[ImageDataset, TabularDataset]:
        """Cria dataset misto"""
        if self.data_type == "image":
            mixed_paths, mixed_labels = self.mixer.mix_image_data(
                self.real_paths,
                self.real_labels,
                self.synthetic_paths,
                self.synthetic_labels,
                proportion,
                balance_classes
            )

            from datatunner.utils.data_loader import DataLoader as DTDataLoader
            transforms = DTDataLoader.get_image_transforms(
                image_size=(224, 224),
                augment=False,
                normalize=True
            )

            return ImageDataset(mixed_paths, mixed_labels, transform=transforms)

        elif self.data_type == "tabular":
            synthetic_data = getattr(self, 'synthetic_data', None)
            synthetic_labels = getattr(self, 'synthetic_labels', None)

            if synthetic_data is None:
                mixed_data, mixed_labels = self.real_data.copy(), self.real_labels.copy()
            else:
                mixed_data, mixed_labels = self.mixer.mix_tabular_data(
                    self.real_data,
                    self.real_labels,
                    synthetic_data,
                    synthetic_labels,
                    proportion,
                    balance_classes
                )

            return TabularDataset(mixed_data, mixed_labels)

        else:
            raise ValueError(f"Tipo de dados não suportado: {self.data_type}")
    
    def _create_test_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Cria test loader"""
        if self.data_type == "image":
            from datatunner.utils.data_loader import DataLoader as DTDataLoader
            transforms = DTDataLoader.get_image_transforms(
                image_size=(224, 224),
                augment=False,
                normalize=True
            )

            test_paths = getattr(self, 'test_paths', None) or self.real_paths
            test_labels = getattr(self, 'test_labels', None) or self.real_labels

            test_dataset = ImageDataset(
                test_paths, test_labels, transform=transforms
            )

            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config['num_workers']
            )

        elif self.data_type == "tabular":
            test_data = getattr(self, 'test_data', None)
            test_labels = getattr(self, 'test_labels', None)

            if test_data is None:
                test_data, test_labels = self.real_data, self.real_labels

            test_dataset = TabularDataset(test_data, test_labels)

            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config['num_workers']
            )

        else:
            raise ValueError(f"Tipo de dados não suportado: {self.data_type}")
    
    def _reset_model(self, model: nn.Module) -> nn.Module:
        """Reinicializa pesos do modelo criando nova instância"""
        try:
            new_model = type(model)()
            return new_model
        except Exception:
            pass

        try:
            import copy
            new_model = copy.deepcopy(model)
            new_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            return new_model
        except Exception:
            pass

        self.logger.warning("Não foi possível reinicializar o modelo. Usando mesma instância.")
        return model
    
    def _aggregate_trial_results(
        self,
        trial_results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Agrega resultados de múltiplos trials"""
        aggregated = {}
        
        # Métricas a agregar
        metrics_to_aggregate = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        
        for metric in metrics_to_aggregate:
            values = [r.get(metric) for r in trial_results if r.get(metric) is not None]
            if values:
                aggregated[metric] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
        
        return aggregated
    
    def _find_best_proportion(self, metric: str = "accuracy"):
        """Encontra melhor proporção baseado em métrica"""
        if not self.results:
            self.best_proportion = None
            self.best_metrics = {}
            self.logger.warning("Nenhum resultado para determinar melhor proporção")
            return

        best_value = -float('inf')
        best_prop = None
        
        for proportion, metrics in self.results.items():
            value = metrics.get(metric)
            if value is not None and value > best_value:
                best_value = value
                best_prop = proportion
        
        if best_prop is None:
            self.best_proportion = list(self.results.keys())[0]
            self.best_metrics = self.results[self.best_proportion]
            self.logger.warning(f"Métrica '{metric}' não encontrada. Usando primeira proporção.")
        else:
            self.best_proportion = best_prop
            self.best_metrics = self.results[best_prop]
    
    def _save_results(self):
        """Salva resultados em JSON"""
        results_file = self.output_dir / "results.json"
        
        results_dict = {
            "experiment_info": {
                "data_type": self.data_type,
                "random_seed": self.random_seed,
                "timestamp": datetime.now().isoformat()
            },
            "results": {str(k): v for k, v in self.results.items()},
            "best_proportion": float(self.best_proportion),
            "best_metrics": self.best_metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        self.logger.info(f"Resultados salvos em: {results_file}")
    
    def plot_results(self, metric: str = "accuracy"):
        """Plota resultados"""
        self.visualizer.plot_proportion_vs_metric(
            self.results,
            metric=metric,
            save_name=f"{metric}_vs_proportion.png"
        )
    
    def plot_multiple_metrics(self, metrics: List[str] = None):
        """Plota múltiplas métricas"""
        self.visualizer.plot_multiple_metrics(
            self.results,
            metrics=metrics,
            save_name="all_metrics.png"
        )
    
    def generate_report(self, model_name: str = None, format: str = "html"):
        """Gera relatório final
        
        Args:
            model_name: Nome do modelo (opcional, auto-detectado se disponível)
            format: Formato do relatório (apenas 'html' suportado)
        """
        if model_name is None and hasattr(self, '_model_name'):
            model_name = self._model_name
        if model_name is None:
            model_name = self.data_type.capitalize()

        experiment_info = {
            "data_type": self.data_type,
            "model_name": model_name,
            "epochs": self.config['epochs'],
            "batch_size": self.config['batch_size']
        }
        
        self.visualizer.generate_summary_report(
            self.results,
            self.best_proportion,
            experiment_info,
            save_name="summary_report.html"
        )
    
    def create_interactive_plot(self, metrics: List[str] = None):
        """Cria gráfico interativo"""
        return self.visualizer.create_interactive_plot(
            self.results,
            metrics=metrics,
            save_name="interactive_plot.html"
        )
