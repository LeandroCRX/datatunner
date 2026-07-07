# DataTunner

**Otimizacao Automatica da Proporcao de Dados Sinteticos para Modelos de Deep Learning**

DataTunner e uma ferramenta Python open-source que automatiza a determinacao da proporcao otima de dados sinteticos para maximizar o desempenho de modelos de machine learning.

## Escopo

**Imagens:** data augmentation (albumentations)
**Dados tabulares:** SMOTE, ADASYN, Borderline-SMOTE, Gaussian Noise, CTGAN, TVAE

## Instalacao

```bash
pip install datatunner
```

## Exemplo Rapido (Dados Tabulares)

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datatunner import DataTunner
from datatunner.models.mlp import MLPClassifier
from datatunner.generators.smote import GaussianNoiseGenerator

# Dados
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Gerar sinteticos
noise_gen = GaussianNoiseGenerator(noise_level=0.15, random_seed=42)
noise_gen.fit(X_train, y_train)
X_synth, y_synth = noise_gen.generate(n_samples=60)

# Otimizar
tunner = DataTunner(data_type='tabular', output_dir='results', random_seed=42,
                    config={'epochs': 10, 'batch_size': 8, 'device': 'cpu'})
tunner.real_data, tunner.real_labels = X_train, y_train
tunner.test_data, tunner.test_labels = X_test, y_test
tunner.synthetic_data, tunner.synthetic_labels = X_synth, y_synth
tunner.class_names = [str(c) for c in np.unique(y_train)]

model = MLPClassifier(input_dim=4, num_classes=3, hidden_layers=[32, 16], dropout=0.2)
results = tunner.optimize(model=model, proportions=[0.0, 0.3, 0.5, 0.7], epochs=10, n_trials=2)
print(f"Melhor proporcao: {results['best_proportion']}")
```

## License

MIT
