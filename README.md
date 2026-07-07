# DataTunner

**Automatic Optimization of Synthetic Data Proportions for Deep Learning Models**

DataTunner is an open-source Python tool that automates the process of determining the optimal proportion of synthetic data to maximize neural network model performance.

## Installation

```bash
pip install datatunner
```

## Quick Start

```python
from datatunner import DataTunner
from datatunner.models.cnn import ResNetClassifier

tunner = DataTunner(
    data_type='image',
    real_data_path='data/real',
    synthetic_data_path='data/synthetic',
    output_dir='results'
)

model = ResNetClassifier(num_classes=10, architecture='resnet18')
results = tunner.optimize(model=model, epochs=50)
print(f"Best proportion: {results['best_proportion']}")
```

## License

MIT
