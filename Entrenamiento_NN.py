import logging
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class NeuralNetworkTrainer:
    """
    Entrenamiento sencillo de una red neuronal (MLP) con interfaz similar a los otros scripts:
    - Carga `outputRecords.csv`
    - Escala características con `StandardScaler`
    - Codifica etiquetas con `LabelEncoder`
    - Optimiza hiperparámetros con GridSearchCV
    - Entrena modelo final y guarda un .pkl con: model, scaler, label_encoder, feature_names, class_names
    """

    def __init__(self, data_file='outputRecords.csv', model_output_file='trainedModel_nn.pkl', params_file='hyperparams_nn.json'):
        self.data_file = data_file
        self.model_output_file = Path(model_output_file)
        self.params_file = Path(params_file)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.class_names = None

        self.analysis_results_dir = Path("analysis_results_nn")
        self.analysis_results_dir.mkdir(exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_and_preprocess(self):
        self.logger.info(f"Cargando datos desde {self.data_file}...")
        try:
            df = pd.read_csv(self.data_file)
        except FileNotFoundError:
            self.logger.error(f"Archivo no encontrado: {self.data_file}")
            return None, None
        except Exception as e:
            self.logger.error(f"Error leyendo {self.data_file}: {e}")
            return None, None

        if df.empty:
            self.logger.error("Dataset vacío.")
            return None, None

        df = df.drop_duplicates()

        if 'activity' not in df.columns:
            self.logger.error("Columna 'activity' no encontrada en el dataset.")
            return None, None

        self.feature_names = [c for c in df.columns if c not in ['timestamp', 'activity']]
        if not self.feature_names:
            self.logger.error("No se encontraron columnas de características.")
            return None, None

        X = df[self.feature_names].values
        y = self.label_encoder.fit_transform(df['activity'].values)
        self.class_names = list(self.label_encoder.classes_)

        unique, counts = np.unique(y, return_counts=True)
        self.logger.info(f"Clases detectadas: {dict(zip(self.class_names, counts))}")

        return X, y

    def optimize_hyperparameters(self, X_train, y_train, cv_folds=3):
        """Grid search over a compact set of MLP hyperparameters (kept small to be fast)."""
        self.logger.info("Iniciando GridSearchCV para MLP...")
        mlp = MLPClassifier(max_iter=500, random_state=42)

        param_grid = {
            'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [1e-4, 1e-3, 1e-2],
            'learning_rate_init': [1e-3, 1e-2]
        }

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid = GridSearchCV(mlp, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        self.logger.info(f"Mejores parámetros MLP: {grid.best_params_}")
        try:
            with open(self.params_file, 'w') as f:
                json.dump({'best_params': grid.best_params_, 'best_score': grid.best_score_}, f, indent=4)
            self.logger.info(f"Parámetros guardados en {self.params_file}")
        except Exception as e:
            self.logger.warning(f"No se pudieron guardar parámetros: {e}")

        return grid.best_params_

    def train_and_evaluate(self, force_optimization=True, test_set_size=0.25):
        X, y = self.load_and_preprocess()
        if X is None:
            return False

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=42, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=42)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        best_params = None
        if force_optimization:
            best_params = self.optimize_hyperparameters(X_train, y_train)
        else:
            if self.params_file.exists():
                try:
                    with open(self.params_file, 'r') as f:
                        cfg = json.load(f)
                        best_params = cfg.get('best_params')
                except Exception:
                    best_params = None

        if best_params is None:
            # default configuration if no optimization
            best_params = {'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'alpha': 0.001, 'learning_rate_init': 0.001}

        # Ensure some sensible training params
        self.model = MLPClassifier(**best_params, max_iter=1000, early_stopping=True, random_state=42)
        self.logger.info(f"Entrenando MLP con: {self.model.get_params()}")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        acc = accuracy_score(y_test, y_pred)
        self.logger.info(f"Accuracy en test: {acc:.4f}")
        self.logger.info(classification_report(y_test_labels, y_pred_labels, zero_division=0))

        # Save simple reports and confusion matrix
        try:
            report_df = pd.DataFrame(classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)).transpose()
            report_df.to_csv(self.analysis_results_dir / 'classification_report_nn.csv')
        except Exception as e:
            self.logger.warning(f"No se pudo guardar el reporte: {e}")

        try:
            cm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.class_names)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix (NN)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(self.analysis_results_dir / 'confusion_matrix_nn.png')
            plt.close()
        except Exception as e:
            self.logger.warning(f"No se pudo generar la matriz de confusión: {e}")

        self._save_trained_model()
        return True

    def _save_trained_model(self):
        if self.model is None:
            self.logger.warning("No hay modelo para guardar.")
            return

        package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }

        try:
            with open(self.model_output_file, 'wb') as f:
                pickle.dump(package, f)
            self.logger.info(f"Modelo guardado en {self.model_output_file}")
        except Exception as e:
            self.logger.error(f"Error guardando modelo: {e}")


def main():
    trainer = NeuralNetworkTrainer()
    success = trainer.train_and_evaluate(force_optimization=False, test_set_size=0.25)
    if success:
        logging.info("Entrenamiento NN completado.")
    else:
        logging.error("Entrenamiento NN falló.")


if __name__ == '__main__':
    main()
