import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 
import pickle
import json
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definición de la clase para manejar el flujo de entrenamiento, optimización y evaluación del modelo.
class ModelTrainer:
    def __init__(self, data_file='outputRecords.csv', params_file='hyperparamsFile.json', model_output_file='trainedModel.pkl'):
        """
        Inicializa la clase con atributos básicos.
        Args:
            data_file (str): Nombre del archivo CSV con los datos.
            params_file (str): Nombre del archivo JSON para guardar/cargar hiperparámetros.
            model_output_file (str): Nombre del archivo .pkl para guardar el modelo entrenado.
        """
        self.data_file = data_file
        self.model_params_file = Path(params_file) # Usar Path para mejor manejo
        self.model_output_file = Path(model_output_file)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  # Codificador para etiquetas categóricas.
        
        self.feature_names = None  # Lista de nombres de las características.
        self.class_names = None    # Nombres de las clases codificadas por LabelEncoder.

        # Crear directorio para resultados de análisis si no existe
        self.analysis_results_dir = Path("analysis_results")
        self.analysis_results_dir.mkdir(exist_ok=True)

    def load_and_preprocess_data(self):
        """
        Carga el dataset desde un archivo CSV, realiza limpieza básica y preprocesamiento.
        Returns:
            tuple (X, y) o (None, None): Características (X) y etiquetas (y) preprocesadas, o None si falla.
        """
        logging.info(f"Cargando datos desde {self.data_file}...")
        try:
            df = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error(f"Error: No se encontró el archivo {self.data_file}.")
            return None, None
        except Exception as e:
            logging.error(f"Error al cargar {self.data_file}: {e}")
            return None, None

        if df.empty:
            logging.warning("El archivo de datos está vacío.")
            return None, None

     
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < initial_rows:
            logging.info(f"Se eliminaron {initial_rows - len(df)} filas duplicadas.")

        if 'activity' not in df.columns:
            logging.error("La columna 'activity' (etiqueta) no se encuentra en el dataset.")
            return None, None

        self.feature_names = [col for col in df.columns if col not in ['timestamp', 'activity']]
        if not self.feature_names:
            logging.error("No se encontraron columnas de características adecuadas en el dataset.")
            return None, None

        logging.info(f"Características identificadas: {self.feature_names}")

        X = df[self.feature_names].values
        
        y_encoded = self.label_encoder.fit_transform(df['activity'].values)
        self.class_names = list(self.label_encoder.classes_) # Guardar nombres de clases
        logging.info(f"Clases detectadas y codificadas: {dict(zip(range(len(self.class_names)), self.class_names))}")
        
        unique_classes, counts = np.unique(y_encoded, return_counts=True)
        min_samples_per_class = 5 # Mínimo para un CV de 5 folds
        if any(counts < min_samples_per_class):
            logging.warning(f"Algunas clases tienen menos de {min_samples_per_class} muestras, lo que podría afectar la validación cruzada.")
            logging.warning(f"Distribución de clases: {dict(zip(self.label_encoder.inverse_transform(unique_classes), counts))}")

        return X, y_encoded

    def optimize_hyperparameters(self, X_train_scaled, y_train):
        """
        Encuentra los mejores hiperparámetros para Random Forest y XGBoost usando GridSearchCV.
        La selección del mejor *tipo* de modelo (RF o XGB) se basa en su rendimiento de CV en el conjunto de entrenamiento.
        Args:
            X_train_scaled: Datos de entrenamiento normalizados.
            y_train: Etiquetas de entrenamiento.
        Returns:
            dict: Diccionario con los mejores parámetros para cada modelo y el modelo recomendado.
        """
        logging.info("Iniciando optimización de hiperparámetros...")

        # Definición del grid de hiperparámetros para Random Forest.
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None] # Para manejar desbalance si existe
        }

        # Definición del grid de hiperparámetros para XGBoost.
        xgb_param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9] # Muestreo de columnas
        }

        # Estrategia de validación cruzada (estratificada para clasificación)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Optimización de Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=cv_strategy, n_jobs=-1, verbose=1, scoring='accuracy')
        rf_grid.fit(X_train_scaled, y_train)
        logging.info(f"Mejores parámetros para Random Forest: {rf_grid.best_params_}")
        logging.info(f"Mejor puntuación CV para Random Forest: {rf_grid.best_score_:.4f}")

        # Optimización de XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss') 
        xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=cv_strategy, n_jobs=-1, verbose=1, scoring='accuracy')
        xgb_grid.fit(X_train_scaled, y_train)
        logging.info(f"Mejores parámetros para XGBoost: {xgb_grid.best_params_}")
        logging.info(f"Mejor puntuación CV para XGBoost: {xgb_grid.best_score_:.4f}")

        # Guardar los parámetros optimizados y el modelo con mejor desempeño CV en entrenamiento
        optimized_params = {
            'random_forest': {
                'params': rf_grid.best_params_,
                'cv_score_on_train': rf_grid.best_score_ 
            },
            'xgboost': {
                'params': xgb_grid.best_params_,
                'cv_score_on_train': xgb_grid.best_score_
            },
            # Selecciona el modelo basado en la puntuación CV del conjunto de entrenamiento
            'best_model_type': 'random_forest' if rf_grid.best_score_ >= xgb_grid.best_score_ else 'xgboost'
        }

        try:
            with open(self.model_params_file, 'w') as f:
                json.dump(optimized_params, f, indent=4)
            logging.info(f"Parámetros optimizados guardados en {self.model_params_file}")
        except IOError as e:
            logging.error(f"No se pudieron guardar los parámetros optimizados: {e}")
            
        return optimized_params

    def train_and_evaluate(self, force_optimization=True, test_set_size=0.2):
        """
        Orquesta el proceso de carga de datos, entrenamiento, optimización y evaluación.
        Args:
            force_optimization (bool): Si es True, se realiza la optimización de hiperparámetros.
                                       Si es False, intenta cargar parámetros guardados.
            test_set_size (float): Proporción del dataset a usar como conjunto de prueba.
        Returns:
            bool: True si el entrenamiento y evaluación fueron exitosos, False en caso contrario.
        """
        X, y = self.load_and_preprocess_data()
        if X is None or y is None:
            logging.error("Fallo en la carga o preprocesamiento de datos. Abortando entrenamiento.")
            return False

        # División del dataset en conjuntos de entrenamiento y prueba
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_set_size, random_state=42, stratify=y
            )
        except ValueError as e:
            logging.error(f"Error en train_test_split (posiblemente por pocas muestras en alguna clase para estratificar): {e}")
            logging.info("Intentando sin estratificación como fallback (no recomendado si hay desbalance).")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_set_size, random_state=42
            )


        # Escalar los datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logging.info("Datos divididos y escalados.")

        # Optimizacion de hiperparametros o carga desde archivo
        optimized_params = None
        if force_optimization:
            optimized_params = self.optimize_hyperparameters(X_train_scaled, y_train)
        else:
            if self.model_params_file.exists():
                try:
                    with open(self.model_params_file, 'r') as f:
                        optimized_params = json.load(f)
                    logging.info(f"Parámetros cargados desde {self.model_params_file}")
                except (IOError, json.JSONDecodeError) as e:
                    logging.warning(f"No se pudieron cargar los parámetros desde {self.model_params_file}: {e}. Realizando optimización.")
                    optimized_params = self.optimize_hyperparameters(X_train_scaled, y_train)
            else:
                logging.info(f"Archivo de parámetros {self.model_params_file} no encontrado. Realizando optimización.")
                optimized_params = self.optimize_hyperparameters(X_train_scaled, y_train)
        
        if not optimized_params:
             logging.error("No se pudieron obtener los hiperparámetros optimizados. Abortando.")
             return False

        # Crear y entrenar el modelo seleccionado con los mejores parámetros
        best_model_type = optimized_params['best_model_type']
        logging.info(f"Modelo seleccionado basado en CV de entrenamiento: {best_model_type}")

        if best_model_type == 'random_forest':
            self.model = RandomForestClassifier(**optimized_params['random_forest']['params'], random_state=42)
        elif best_model_type == 'xgboost':
            xgb_params = optimized_params['xgboost']['params']
            self.model = xgb.XGBClassifier(**xgb_params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        else:
            logging.error(f"Tipo de modelo desconocido: {best_model_type}. Abortando.")
            return False

        logging.info(f"Entrenando modelo final ({best_model_type}) con parámetros: {self.model.get_params()}")
        self.model.fit(X_train_scaled, y_train)

        # Realizar predicciones en el conjunto de prueba
        y_pred = self.model.predict(X_test_scaled)

        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        self.analyze_and_save_results(y_test, y_pred, y_test_labels, y_pred_labels)

        self.save_trained_model()
        
        return True

    def save_trained_model(self):
        """
        Guarda el modelo entrenado, el escalador, el codificador de etiquetas, 
        los nombres de las características y los nombres de las clases.
        """
        if self.model is None:
            logging.warning("No hay modelo entrenado para guardar.")
            return

        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names # Guardar los nombres de las clases
        }
        
        try:
            with open(self.model_output_file, 'wb') as f:
                pickle.dump(model_package, f)
            logging.info(f"Modelo, escalador, codificador y metadatos guardados en {self.model_output_file}")
        except IOError as e:
            logging.error(f"Error al guardar el modelo: {e}")

    def analyze_and_save_results(self, y_test_encoded, y_pred_encoded, y_test_labels, y_pred_labels):
        """
        Analiza y visualiza los resultados del modelo, guardando los artefactos.
        Args:
            y_test_encoded: Etiquetas reales (codificadas) del conjunto de prueba.
            y_pred_encoded: Etiquetas predichas (codificadas) por el modelo.
            y_test_labels: Etiquetas reales (descodificadas).
            y_pred_labels: Etiquetas predichas (descodificadas).
        """
        logging.info("Analizando resultados del modelo...")

        # 1. Métricas generales
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)
        
        logging.info("\n--- Métricas de Evaluación en Conjunto de Prueba ---")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info("\nInforme de Clasificación Detallado:")
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))


        report_df = pd.DataFrame(report_dict).transpose()
        try:
            report_df.to_csv(self.analysis_results_dir / 'classification_metrics.csv')
            logging.info(f"Métricas de clasificación guardadas en {self.analysis_results_dir / 'classification_metrics.csv'}")
        except IOError as e:
            logging.error(f"No se pudieron guardar las métricas de clasificación: {e}")

        # 2. Matriz de confusión
        plt.figure(figsize=(18, 7))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.class_names) # Usar class_names para orden consistente
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Matriz de Confusión (Valores Absolutos)')
        plt.xlabel('Predicción')
        plt.ylabel('Real')

        plt.subplot(1, 2, 2)
        cm_norm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.class_names, normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Matriz de Confusión (Normalizada)')
        plt.xlabel('Predicción')
        plt.ylabel('Real')

        plt.tight_layout()
        try:
            plt.savefig(self.analysis_results_dir / 'confusion_matrices.png')
            logging.info(f"Matrices de confusión guardadas en {self.analysis_results_dir / 'confusion_matrices.png'}")
        except IOError as e:
            logging.error(f"No se pudieron guardar las matrices de confusión: {e}")
        plt.close()

        # 3. Gráfico de barras para Precision, Recall y F1-score por clase
        metrics_per_class = {cls: report_dict[cls] for cls in self.class_names if cls in report_dict}
        metrics_df = pd.DataFrame(metrics_per_class).T[['precision', 'recall', 'f1-score']]
        
        plt.figure(figsize=(12, 7))
        metrics_df.plot(kind='bar', width=0.8, ax=plt.gca())
        plt.title('Métricas por Clase (Precision, Recall, F1-score)')
        plt.xlabel('Clase')
        plt.ylabel('Puntuación')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        try:
            plt.savefig(self.analysis_results_dir / 'metrics_by_class.png')
            logging.info(f"Gráfico de métricas por clase guardado en {self.analysis_results_dir / 'metrics_by_class.png'}")
        except IOError as e:
            logging.error(f"No se pudo guardar el gráfico de métricas por clase: {e}")
        plt.close()

        # 4. Importancia de características
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, max(8, len(self.feature_names) // 2))) # Ajustar altura dinámicamente
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette="viridis") # Top 20
            plt.title(f'Importancia de Características ({type(self.model).__name__})')
            plt.tight_layout()
            try:
                plt.savefig(self.analysis_results_dir / 'feature_importance.png')
                logging.info(f"Gráfico de importancia de características guardado en {self.analysis_results_dir / 'feature_importance.png'}")
            except IOError as e:
                logging.error(f"No se pudo guardar el gráfico de importancia de características: {e}")
            plt.close()
        else:
            logging.info(f"El modelo {type(self.model).__name__} no tiene el atributo 'feature_importances_'.")


# Función principal para ejecutar el flujo
def main():
    """
    Función principal que instancia ModelTrainer y ejecuta el proceso de entrenamiento y evaluación.
    """
    logging.info("=== Iniciando Proceso de Entrenamiento y Evaluación del Modelo ===")
    
    trainer = ModelTrainer(
        data_file='outputRecords.csv', 
        params_file='hyperparamsFile.json', 
        model_output_file='trainedModel.pkl'
    )
    
    success = trainer.train_and_evaluate(force_optimization=True, test_set_size=0.25)

    if success:
        logging.info("=== Proceso de Entrenamiento y Evaluación Completado Exitosamente ===")
    else:
        logging.error("=== Proceso de Entrenamiento y Evaluación Fallido ===")

if __name__ == "__main__":
    main()