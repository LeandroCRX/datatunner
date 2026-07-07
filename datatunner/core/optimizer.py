"""
Motor principal de otimizacao do DataTunner
"""

import json
import random
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
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


_VALID_DATA_TYPES = ("image", "tabular")


class DataTunner:
    """
    Motor principal de otimizacao de proporcoes de dados sinteticos
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
            synthetic_data_path: Caminho para dados sinteticos
            test_data_path: Caminho para dados de teste (opcional)
            output_dir: Diretorio de saida
            random_seed: Seed para reprodutibilidade
            config: Configuracoes customizadas
        """
        self.data_type = data_type.lower()
        if self.data_type not in _VALID_DATA_TYPES:
            raise ValueError(
                f"Tipo de dados invalido: {self.data_type}. "
                f"Valores suportados: {_VALID_DATA_TYPES}"
            )

        self.real_data_path = real_data_path
        self.synthetic_data_path = synthetic_data_path
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed

        self.config = get_config(config)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_directories(str(self.output_dir))

        self.mixer = DataMixer(random_seed=random_seed)
        self.evaluator = None
        self.visualizer = ResultsVisualizer(
            output_dir=str(self.output_dir),
            style=self.config.get('plot_style', 'auto')
        )

        self.real_data = None
        self.real_labels = None
        self.synthetic_data = None
        self.synthetic_labels = None
        self.test_data = None
        self.test_labels = None
        self.class_names = None

        self.real_paths = None
        self.real_labels = None
        self.synthetic_paths = None
        self.synthetic_labels = None
        self.test_paths = None
        self.test_labels = None

        self.results = {}
        self.best_proportion = None
        self.best_metrics = None
        self._model_name = None

        self._set_seeds()
        self._setup_logging()

        if real_data_path:
            self._load_data()

    def _set_seeds(self):
        """Configura seeds para reprodutibilidade"""
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

        logger = logging.getLogger("datatunner")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            formatter = logging.Formatter(self.config['log_format'])
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        self.logger = logger
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
            raise ValueError(f"Tipo de dados nao suportado: {self.data_type}")

    def _load_image_data(self):
        """Carrega dados de imagem"""
        loader = DTDataLoader()

        self.real_paths, self.real_labels, self.class_names = loader.load_image_data(
            self.real_data_path
        )

        self.synthetic_paths = []
        self.synthetic_labels = []
        if self.synthetic_data_path:
            self.synthetic_paths, self.synthetic_labels, _ = loader.load_image_data(
                self.synthetic_data_path
            )

        self.test_paths = None
        self.test_labels = None
        if self.test_data_path:
            self.test_paths, self.test_labels, _ = loader.load_image_data(
                self.test_data_path
            )

        self.logger.info(f"Classes detectadas: {self.class_names}")
        self.logger.info(f"Dados reais: {len(self.real_paths)} imagens")
        if self.synthetic_paths:
            self.logger.info(f"Dados sinteticos: {len(self.synthetic_paths)} imagens")

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

        self.synthetic_data = None
        self.synthetic_labels = None
        if self.synthetic_data_path:
            df_syn = pd.read_csv(self.synthetic_data_path)
            if 'target' in df_syn.columns:
                self.synthetic_data = df_syn.drop(columns=['target']).values
                self.synthetic_labels = df_syn['target'].values
                self.logger.info(f"Dados sinteticos: {len(self.synthetic_data)} amostras")
            else:
                raise ValueError(
                    "Arquivo sintetico deve conter a coluna 'target'. "
                    "Encontrado apenas: " + str(list(df_syn.columns))
                )

        self.class_names = [str(c) for c in np.unique(self.real_labels)]
        self.logger.info(f"Classes detectadas: {self.class_names}")
        self.logger.info(f"Dados reais: {len(self.real_data)} amostras")

    def optimize(
        self,
        model: Optional[nn.Module] = None,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        proportions: Optional[List[float]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        n_trials: int = 1,
        balance_classes: bool = True,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Executa otimizacao para encontrar melhor proporcao

        Args:
            model: Instancia do modelo PyTorch (usada como template)
            model_factory: Funcao callable que retorna nova instancia do modelo
                          a cada chamada. Recomendado para reprodutibilidade.
            proportions: Lista de proporcoes a testar
            epochs: Numero de epocas
            batch_size: Tamanho do batch
            learning_rate: Taxa de aprendizado
            n_trials: Numero de repeticoes por proporcao
            balance_classes: Se deve balancear classes
            task_type: Tipo de tarefa ('classification' ou 'regression')

        Returns:
            Dicionario com resultados
        """
        if model is None and model_factory is None:
            raise ValueError("Forneça 'model' ou 'model_factory'")

        if proportions is None:
            proportions = self.config['proportions']

        if not all(0.0 <= p <= 1.0 for p in proportions):
            raise ValueError("Todas as proporcoes devem estar entre 0.0 e 1.0")

        if model is not None:
            self._model_name = type(model).__name__
        elif model_factory is not None:
            try:
                self._model_name = model_factory.__name__
            except AttributeError:
                try:
                    self._model_name = type(model_factory()).__name__
                except Exception:
                    self._model_name = "CustomModel"

        self.logger.info(f"Iniciando otimizacao com proporcoes: {proportions}")
        self.logger.info(f"Epocas: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        self.logger.info(f"Modelo: {self._model_name}, Task: {task_type}")

        device = self.config['device']
        self.evaluator = ModelEvaluator(device=device, task_type=task_type)

        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        results = {}

        for proportion in proportions:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testando proporcao: {proportion:.1%}")
            self.logger.info(f"{'='*60}")

            trial_results = []

            for trial in range(n_trials):
                self.logger.info(f"\nTrial {trial + 1}/{n_trials}")

                random.seed(self.random_seed + trial)
                np.random.seed(self.random_seed + trial)
                torch.manual_seed(self.random_seed + trial)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_seed + trial)

                mixed_dataset = self._create_mixed_dataset(
                    proportion, balance_classes
                )

                train_loader, val_loader = create_data_loaders(
                    mixed_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=self.config['num_workers'],
                    validation_split=self.config['validation_split'],
                    random_state=self.random_seed + trial
                )

                test_loader = self._create_test_loader(batch_size)

                if model_factory is not None:
                    model_instance = model_factory()
                    if not isinstance(model_instance, nn.Module):
                        raise TypeError("model_factory deve retornar uma instancia de nn.Module")
                else:
                    model_instance = self._reset_model(model)

                optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)

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

                acc = test_metrics.get('accuracy', 0.0)
                f1 = test_metrics.get('f1_score', 0.0)
                self.logger.info(f"Accuracy: {float(acc):.4f}")
                self.logger.info(f"F1-Score: {float(f1):.4f}")

            aggregated_metrics = self._aggregate_trial_results(trial_results)
            results[proportion] = aggregated_metrics

            avg_acc = aggregated_metrics.get('accuracy', 0.0)
            self.logger.info(f"\nMedia - Accuracy: {float(avg_acc):.4f}")

        self.results = results

        self._find_best_proportion()

        self._save_results()

        best_acc = self.best_metrics.get('accuracy', 0.0) if self.best_metrics else 0.0
        self.logger.info(f"\n{'='*60}")
        if self.best_proportion is not None:
            self.logger.info(f"MELHOR PROPORCAO: {self.best_proportion:.1%}")
            self.logger.info(f"MELHOR ACCURACY: {float(best_acc):.4f}")
        else:
            self.logger.info("Nenhum resultado encontrado.")
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

            transforms = DTDataLoader.get_image_transforms(
                image_size=(224, 224),
                augment=False,
                normalize=True
            )

            return ImageDataset(mixed_paths, mixed_labels, transform=transforms)

        elif self.data_type == "tabular":
            synthetic_data = self.synthetic_data
            synthetic_labels = self.synthetic_labels

            if synthetic_data is None:
                mixed_data = self.real_data.copy()
                mixed_labels = self.real_labels.copy()
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
            raise ValueError(f"Tipo de dados nao suportado: {self.data_type}")

    def _create_test_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Cria test loader"""
        if self.data_type == "image":
            transforms = DTDataLoader.get_image_transforms(
                image_size=(224, 224),
                augment=False,
                normalize=True
            )

            test_paths = self.test_paths if self.test_paths is not None else self.real_paths
            test_labels = self.test_labels if self.test_labels is not None else self.real_labels

            test_dataset = ImageDataset(test_paths, test_labels, transform=transforms)

            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config['num_workers']
            )

        elif self.data_type == "tabular":
            test_data = self.test_data if self.test_data is not None else self.real_data
            test_labels = self.test_labels if self.test_labels is not None else self.real_labels

            test_dataset = TabularDataset(test_data, test_labels)

            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config['num_workers']
            )

        else:
            raise ValueError(f"Tipo de dados nao suportado: {self.data_type}")

    def _reset_model(self, model: nn.Module) -> nn.Module:
        """Reinicializa pesos do modelo criando nova instancia"""
        try:
            new_model = type(model)()
            return new_model
        except Exception as e:
            self.logger.warning(
                f"Nao foi possivel recriar via type(model)(): {e}. "
                "Forneca model_factory para reprodutibilidade. Tentando deepcopy..."
            )

        try:
            new_model = copy.deepcopy(model)
            new_model.apply(
                lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None
            )
            return new_model
        except Exception as e:
            self.logger.warning(
                f"Deepcopy + reset_parameters falhou: {e}. "
                "ATENCAO: reutilizando modelo sem reinicializar - trials compartilharao pesos."
            )
            return model

    def _aggregate_trial_results(
        self,
        trial_results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Agrega resultados de multiplos trials"""
        aggregated = {}

        metrics_to_aggregate = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        for metric in metrics_to_aggregate:
            values = [r.get(metric) for r in trial_results if r.get(metric) is not None]
            if values:
                aggregated[metric] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))

        return aggregated

    def _find_best_proportion(self, metric: str = "accuracy"):
        """Encontra melhor proporcao baseado em metrica"""
        if not self.results:
            self.best_proportion = None
            self.best_metrics = {}
            self.logger.warning("Nenhum resultado para determinar melhor proporcao")
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
            self.logger.warning(f"Metrica '{metric}' nao encontrada. Usando primeira proporcao.")
        else:
            self.best_proportion = best_prop
            self.best_metrics = self.results[best_prop]

    def _save_results(self):
        """Salva resultados em JSON"""
        results_file = self.output_dir / "results.json"

        best_prop_value = float(self.best_proportion) if self.best_proportion is not None else None

        results_dict = {
            "experiment_info": {
                "data_type": self.data_type,
                "model_name": self._model_name,
                "random_seed": self.random_seed,
                "timestamp": datetime.now().isoformat()
            },
            "results": {str(k): v for k, v in self.results.items()},
            "best_proportion": best_prop_value,
            "best_metrics": self.best_metrics
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)

        self.logger.info(f"Resultados salvos em: {results_file}")

    def plot_results(self, metric: str = "accuracy"):
        """Plota resultados"""
        self.visualizer.plot_proportion_vs_metric(
            self.results,
            metric=metric,
            save_name=f"{metric}_vs_proportion.png"
        )

    def plot_multiple_metrics(self, metrics: Optional[List[str]] = None):
        """Plota multiplas metricas"""
        self.visualizer.plot_multiple_metrics(
            self.results,
            metrics=metrics,
            save_name="all_metrics.png"
        )

    def generate_report(self, model_name: Optional[str] = None, format: str = "html"):
        """Gera relatorio final

        Args:
            model_name: Nome do modelo (opcional, auto-detectado se disponivel)
            format: Formato do relatorio (apenas 'html' suportado)
        """
        if model_name is None and self._model_name is not None:
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

    def create_interactive_plot(self, metrics: Optional[List[str]] = None):
        """Cria grafico interativo"""
        return self.visualizer.create_interactive_plot(
            self.results,
            metrics=metrics,
            save_name="interactive_plot.html"
        )
