"""
Sistema de avaliacao de modelos
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
from tqdm import tqdm

from datatunner.utils.metrics import compute_metrics


class ModelEvaluator:
    """Avaliador de performance de modelos"""

    def __init__(
        self,
        device: str = "cuda",
        task_type: str = "classification"
    ):
        """
        Args:
            device: Dispositivo (cuda/cpu/mps)
            task_type: Tipo de tarefa (classification/regression)
        """
        self.task_type = task_type
        self.device = self._resolve_device(device)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve o dispositivo respeitando a escolha do usuario"""
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        class_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Avalia modelo em conjunto de teste

        Args:
            model: Modelo PyTorch
            test_loader: DataLoader de teste
            class_names: Nomes das classes (para classificacao)

        Returns:
            Dicionario com metricas
        """
        model.eval()
        model.to(self.device)

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Avaliando"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)

                if self.task_type == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                else:
                    all_preds.append(outputs.cpu().numpy())

                all_labels.append(labels.cpu().numpy())

        if self.task_type == "classification":
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            metrics = compute_metrics(
                all_labels,
                all_preds,
                task_type="classification",
                y_pred_proba=all_probs,
                class_names=class_names
            )
        else:
            all_preds = np.concatenate(all_preds).ravel()
            all_labels = np.concatenate(all_labels).ravel()
            metrics = compute_metrics(
                all_labels,
                all_preds,
                task_type="regression"
            )

        return metrics

    def train_and_evaluate(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_path: Optional[str] = None,
        class_names: Optional[list] = None
    ) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """
        Treina e avalia modelo

        Args:
            model: Modelo PyTorch
            train_loader: DataLoader de treino
            val_loader: DataLoader de validacao
            test_loader: DataLoader de teste
            optimizer: Otimizador
            criterion: Funcao de perda
            epochs: Numero de epocas
            early_stopping_patience: Paciencia para early stopping
            checkpoint_path: Caminho para salvar checkpoint
            class_names: Nomes das classes

        Returns:
            Tupla (test_metrics, training_history)
        """
        model.to(self.device)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        checkpoint_saved = False

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion
            )

            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"Epoca [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)
                    checkpoint_saved = True
            else:
                patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping na epoca {epoch+1}")
                    break

        if checkpoint_path and checkpoint_saved:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            )

        test_metrics = self.evaluate_model(model, test_loader, class_names)

        return test_metrics, history

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Treina uma epoca"""
        model.train()

        running_loss = 0.0
        running_corrects = 0
        n_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if self.task_type == "classification":
                _, preds = torch.max(outputs, 1)
                running_corrects += int(torch.sum(preds == labels.data).item())

            n_samples += inputs.size(0)

        if n_samples == 0:
            return 0.0, 0.0

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects / n_samples if self.task_type == "classification" else 0.0

        return epoch_loss, float(epoch_acc)

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Valida uma epoca"""
        model.eval()

        running_loss = 0.0
        running_corrects = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                if self.task_type == "classification":
                    _, preds = torch.max(outputs, 1)
                    running_corrects += int(torch.sum(preds == labels.data).item())

                n_samples += inputs.size(0)

        if n_samples == 0:
            return 0.0, 0.0

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects / n_samples if self.task_type == "classification" else 0.0

        return epoch_loss, float(epoch_acc)
