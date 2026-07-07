# ============================================
# DataTunner - Teste no Google Colab com Iris
# ============================================
# Copie e cole este arquivo inteiro no Colab
# ou execute cada celula individualmente abaixo
# ============================================

# --- CELULA 1: Instalacao ---
# !pip install datatunner
# !pip install torch torchvision matplotlib seaborn scikit-learn pandas tqdm plotly albumentations imbalanced-learn

# --- CELULA 2: Codigo principal ---
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from datatunner import DataTunner
# from datatunner.models.mlp import MLPClassifier
# from datatunner.generators.smote import GaussianNoiseGenerator

# iris = load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# noise_gen = GaussianNoiseGenerator(noise_level=0.15, random_seed=42)
# noise_gen.fit(X_train, y_train)
# X_synth, y_synth = noise_gen.generate(n_samples=60)

# tunner = DataTunner(data_type='tabular', output_dir='results_colab', random_seed=42,
#                     config={'epochs': 10, 'batch_size': 8, 'validation_split': 0.2, 'device': 'cpu', 'num_workers': 0})
# tunner.real_data = X_train
# tunner.real_labels = y_train
# tunner.test_data = X_test
# tunner.test_labels = y_test
# tunner.synthetic_data = X_synth
# tunner.synthetic_labels = y_synth
# tunner.class_names = [str(c) for c in np.unique(y_train)]

# model = MLPClassifier(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)), hidden_layers=[32, 16], dropout=0.2)
# results = tunner.optimize(model=model, proportions=[0.0, 0.3, 0.5, 0.7], epochs=10, batch_size=8, learning_rate=0.01, n_trials=2)

# print(f"Melhor proporcao: {results['best_proportion']}")
# for prop, metrics in results['results'].items():
#     print(f"  prop={prop:.1f}: acc={float(metrics.get('accuracy', 0)):.4f}")
# tunner.plot_results()
# tunner.plot_multiple_metrics()
