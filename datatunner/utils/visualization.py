"""
Visualização de resultados e geração de gráficos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import plotly.graph_objects as go


class ResultsVisualizer:
    """Visualizador de resultados do DataTunner"""
    
    def __init__(self, output_dir: str = "results", style: str = "auto"):
        """
        Args:
            output_dir: Diretório para salvar gráficos
            style: Estilo dos gráficos matplotlib ('auto' detecta o melhor disponível)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Selecionar estilo de forma robusta (compatível com qualquer versão do matplotlib)
        self._apply_plot_style(style)
        sns.set_palette("husl")

    def _apply_plot_style(self, style: str):
        """Aplica estilo de grafico de forma segura"""
        available_styles = plt.style.available

        if style != "auto" and style in available_styles:
            try:
                plt.style.use(style)
                return
            except Exception:
                pass

        fallbacks = ["seaborn-v0_8", "seaborn", "ggplot", "bmh"]
        for s in fallbacks:
            if s in available_styles:
                try:
                    plt.style.use(s)
                    return
                except Exception:
                    continue

        try:
            plt.style.use("default")
        except Exception:
            pass
    
    def plot_proportion_vs_metric(
        self,
        results: Dict[float, Dict[str, float]],
        metric: str = "accuracy",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plota métrica vs proporção de dados sintéticos
        
        Args:
            results: Dicionário {proportion: {metric: value}}
            metric: Nome da métrica a plotar
            save_name: Nome do arquivo para salvar
        """
        proportions = sorted(results.keys())
        values = [results[p].get(metric) for p in proportions]

        valid = [(p, v) for p, v in zip(proportions, values) if v is not None]
        if not valid:
            print(f"Nenhum valor valido para a metrica '{metric}'")
            return

        valid_props, valid_values = zip(*valid)

        plt.figure(figsize=(12, 6))
        plt.plot(valid_props, valid_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Proporcao de Dados Sinteticos', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'{metric.capitalize()} vs Proporcao de Dados Sinteticos', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(valid_props)

        best_idx = int(np.argmax(valid_values))
        plt.scatter(
            [valid_props[best_idx]],
            [valid_values[best_idx]],
            color='red',
            s=200,
            marker='*',
            zorder=5,
            label=f'Melhor: {valid_props[best_idx]:.1%}'
        )
        plt.legend()

        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_multiple_metrics(
        self,
        results: Dict[float, Dict[str, float]],
        metrics: List[str] = None,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plota múltiplas métricas em um único gráfico
        
        Args:
            results: Dicionário {proportion: {metric: value}}
            metrics: Lista de métricas a plotar
            save_name: Nome do arquivo para salvar
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        proportions = sorted(results.keys())
        
        plt.figure(figsize=(14, 8))
        
        for metric in metrics:
            values = [results[p].get(metric, None) for p in proportions]
            if all(v is not None for v in values):
                plt.plot(proportions, values, marker='o', label=metric.capitalize(), linewidth=2)
        
        plt.xlabel('Proporção de Dados Sintéticos', fontsize=12)
        plt.ylabel('Valor da Métrica', fontsize=12)
        plt.title('Métricas vs Proporção de Dados Sintéticos', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(proportions)
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plota matriz de confusão
        
        Args:
            conf_matrix: Matriz de confusão
            class_names: Nomes das classes
            save_name: Nome do arquivo para salvar
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.xlabel('Predição', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        plt.title('Matriz de Confusão', fontsize=14)
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ) -> None:
        """
        Plota histórico de treinamento
        
        Args:
            history: Dicionário com loss e métricas por época
            save_name: Nome do arquivo para salvar
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Histórico de Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Métrica (accuracy)
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Histórico de Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plot(
        self,
        results: Dict[float, Dict[str, float]],
        metrics: List[str] = None,
        save_name: Optional[str] = None
    ) -> go.Figure:
        """
        Cria gráfico interativo com Plotly
        
        Args:
            results: Dicionário {proportion: {metric: value}}
            metrics: Lista de métricas a plotar
            save_name: Nome do arquivo HTML para salvar
            
        Returns:
            Figura Plotly
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        proportions = sorted(results.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[p].get(metric, None) for p in proportions]
            if all(v is not None for v in values):
                fig.add_trace(go.Scatter(
                    x=proportions,
                    y=values,
                    mode='lines+markers',
                    name=metric.capitalize(),
                    hovertemplate=f'{metric.capitalize()}: %{{y:.4f}}<br>Proporção: %{{x:.1%}}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Métricas vs Proporção de Dados Sintéticos',
            xaxis_title='Proporção de Dados Sintéticos',
            yaxis_title='Valor da Métrica',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        if save_name:
            fig.write_html(str(self.output_dir / save_name))
        
        return fig
    
    def generate_summary_report(
        self,
        results: Dict[float, Dict[str, float]],
        best_proportion: float,
        experiment_info: Dict,
        save_name: str = "summary_report.html"
    ) -> None:
        """
        Gera relatório HTML resumido
        
        Args:
            results: Resultados dos experimentos
            best_proportion: Melhor proporção encontrada
            experiment_info: Informações do experimento
            save_name: Nome do arquivo HTML
        """
        best_prop_str = f"{best_proportion:.1%}" if best_proportion is not None else "N/A"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DataTunner - Relatorio de Resultados</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .metric-box {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .highlight {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    text-align: center;
                    font-size: 1.2em;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>DataTunner - Relatorio de Resultados</h1>

                <div class="highlight">
                    <strong>Melhor Proporcao de Dados Sinteticos:</strong> {best_prop_str}
                </div>

                <h2>Informacoes do Experimento</h2>
                <div class="metric-box">
                    <p><strong>Tipo de Dados:</strong> {experiment_info.get('data_type', 'N/A')}</p>
                    <p><strong>Modelo:</strong> {experiment_info.get('model_name', 'N/A')}</p>
                    <p><strong>Epocas:</strong> {experiment_info.get('epochs', 'N/A')}</p>
                    <p><strong>Batch Size:</strong> {experiment_info.get('batch_size', 'N/A')}</p>
                </div>

                <h2>Resultados por Proporcao</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Proporcao</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for proportion in sorted(results.keys()):
            metrics = results[proportion]
            acc = metrics.get('accuracy') or 0
            prec = metrics.get('precision') or 0
            rec = metrics.get('recall') or 0
            f1 = metrics.get('f1_score') or 0
            html_content += f"""
                        <tr>
                            <td>{proportion:.1%}</td>
                            <td>{float(acc):.4f}</td>
                            <td>{float(prec):.4f}</td>
                            <td>{float(rec):.4f}</td>
                            <td>{float(f1):.4f}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
                
                <h2>💡 Recomendações</h2>
                <div class="metric-box">
                    <p>✅ Use a proporção ótima identificada para maximizar a performance do modelo</p>
                    <p>✅ Considere realizar validação cruzada para maior robustez</p>
                    <p>✅ Avalie o trade-off entre performance e custo computacional</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / save_name, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Relatório salvo em: {self.output_dir / save_name}")


def plot_results(
    results: Dict[float, Dict[str, float]],
    metric: str = "accuracy",
    output_dir: str = "results"
) -> None:
    """
    Função de conveniência para plotar resultados
    
    Args:
        results: Resultados dos experimentos
        metric: Métrica a plotar
        output_dir: Diretório de saída
    """
    visualizer = ResultsVisualizer(output_dir=output_dir)
    visualizer.plot_proportion_vs_metric(
        results, metric=metric, save_name=f"{metric}_vs_proportion.png"
    )
