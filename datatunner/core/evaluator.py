"""
Sistema de avaliação de modelos
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
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
            device: Dispositivo (cuda/cpu)
            task_type: Tipo de tarefa (classification/regression)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task_type = task_type
    
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
            class_names: Nomes das classes (para classificação)
            
        Returns:
            Dicionário com métricas
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
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                else:
                    all_preds.extend(outputs.cpu().numpy())
                
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calcular métricas
        if self.task_type == "classification":
            all_probs = np.array(all_probs)
            metrics = compute_metrics(
                all_labels,
                all_preds,
                task_type="classification",
                y_pred_proba=all_probs,
                class_names=class_names
            )
        else:
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
    ) -> Tuple[Dict[str, Any], Dict[str, list]]:
        """
        Treina e avalia modelo
        
        Args:
            model: Modelo PyTorch
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            test_loader: DataLoader de teste
            optimizer: Otimizador
            criterion: Função de perda
            epochs: Número de épocas
            early_stopping_patience: Paciência para early stopping
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
        
        for epoch in range(epochs):
            # Treino
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion
            )
            
            # Validação
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion
            )
            
            # Registrar histórico
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Época [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Salvar melhor modelo
                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping na época {epoch+1}")
                    break
        
        # Carregar melhor modelo
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path))
        
        # Avaliar no conjunto de teste
        test_metrics = self.evaluate_model(model, test_loader, class_names)
        
        return test_metrics, history
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Treina uma época
        
        Returns:
            Tupla (loss, accuracy)
        """
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
                running_corrects += torch.sum(preds == labels.data)
            
            n_samples += inputs.size(0)
        
        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples if self.task_type == "classification" else 0.0
        
        return epoch_loss, float(epoch_acc)
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Valida uma época
        
        Returns:
            Tupla (loss, accuracy)
        """
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
                    running_corrects += torch.sum(preds == labels.data)
                
                n_samples += inputs.size(0)
        
        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples if self.task_type == "classification" else 0.0
        
        return epoch_loss, float(epoch_acc)
