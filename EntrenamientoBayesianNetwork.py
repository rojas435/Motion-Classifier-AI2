# --- START OF FILE AdvancedHyperparameterTuner.py ---

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

class AutomatedModelPipeline:

    def __init__(self, dataset_path: str = 'outputRecords.csv',
                 output_directory: str = "pipeline_outputs",
                 model_artifact_name: str = 'best_pose_classifier.pkl',
                 config_name: str = 'optimal_hyperparameters.json'):

        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_directory)
        self.artifacts_dir = self.output_dir / "artifacts"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.reports_dir = self.output_dir / "reports"

        self.model_artifact_path = self.artifacts_dir / model_artifact_name
        self.config_path = self.artifacts_dir / config_name

        self.data_scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.trained_classifier = None
        self.dataset_column_names = []

        self._initialize_directories()
        self._configure_logging()

    def _configure_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "pipeline_log.txt", mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _ingest_and_prepare_data(self):
        self.logger.info(f"Attempting to load dataset from: {self.dataset_path}")
        try:
            raw_df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            self.logger.error(f"Dataset not found at {self.dataset_path}.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
        
        self.logger.info(f"Dataset loaded. Shape: {raw_df.shape}")
        
        original_rows = len(raw_df)
        processed_df = raw_df.drop_duplicates()
        if len(processed_df) < original_rows:
            self.logger.info(f"Removed {original_rows - len(processed_df)} duplicate rows.")

        if 'activity' not in processed_df.columns:
            self.logger.error("Target column 'activity' not found in dataset.")
            raise ValueError("Missing target column 'activity'.")

        self.dataset_column_names = [col for col in processed_df.columns if col not in ['timestamp', 'activity']]
        if not self.dataset_column_names:
            self.logger.error("No feature columns identified after excluding 'timestamp' and 'activity'.")
            raise ValueError("No feature columns available.")

        features = processed_df[self.dataset_column_names].values
        encoded_target = self.target_encoder.fit_transform(processed_df['activity'].values)
        self.logger.info(f"Data prepared. Features shape: {features.shape}, Target shape: {encoded_target.shape}")
        self.logger.info(f"Target classes: {list(self.target_encoder.classes_)}")
        
        return features, encoded_target

    def _objective_rf(self, trial, X, y):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 35, log=False) if trial.suggest_categorical('use_max_depth', [True, False]) else None,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=trial.number)
        score = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy').mean()
        return score

    def _objective_xgb(self, trial, X, y):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1 reg
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2 reg
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss', # mlogloss for multiclass
            'n_jobs': -1
        }
        model = xgb.XGBClassifier(**params)
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=trial.number)
        score = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy').mean()
        return score

    def _find_optimal_configurations(self, X_train, y_train, n_trials_optuna=30):
        self.logger.info(f"Starting hyperparameter optimization with Optuna ({n_trials_optuna} trials per model).")
        
        study_rf = optuna.create_study(direction='maximize', study_name='RandomForest_Optimization')
        study_rf.optimize(lambda trial: self._objective_rf(trial, X_train, y_train), n_trials=n_trials_optuna, show_progress_bar=True)

        study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost_Optimization')
        study_xgb.optimize(lambda trial: self._objective_xgb(trial, X_train, y_train), n_trials=n_trials_optuna, show_progress_bar=True)

        rf_config = {'params': study_rf.best_params, 'score': study_rf.best_value}
        xgb_config = {'params': study_xgb.best_params, 'score': study_xgb.best_value}

        self.logger.info(f"Random Forest - Best CV Score: {rf_config['score']:.4f} with params: {rf_config['params']}")
        self.logger.info(f"XGBoost - Best CV Score: {xgb_config['score']:.4f} with params: {xgb_config['params']}")

        chosen_model_type = 'random_forest' if rf_config['score'] >= xgb_config['score'] else 'xgboost'
        self.logger.info(f"Recommended model type based on CV score: {chosen_model_type.upper()}")

        optimal_configurations = {
            'random_forest_config': rf_config,
            'xgboost_config': xgb_config,
            'selected_model_type': chosen_model_type
        }

        try:
            with open(self.config_path, 'w') as f:
                json.dump(optimal_configurations, f, indent=4)
            self.logger.info(f"Optimal configurations saved to {self.config_path}")
        except IOError as e:
            self.logger.error(f"Failed to save optimal configurations: {e}")

        self._generate_optuna_visualizations(study_rf, "RandomForest")
        self._generate_optuna_visualizations(study_xgb, "XGBoost")
            
        return optimal_configurations

    def _generate_optuna_visualizations(self, study, model_name_prefix):
        try:
            fig_history = plot_optimization_history(study)
            fig_history.write_image(self.visualizations_dir / f"{model_name_prefix}_optuna_history.png")
            
            fig_importance = plot_param_importances(study)
            fig_importance.write_image(self.visualizations_dir / f"{model_name_prefix}_optuna_param_importance.png")
            
            # Slice plot puede ser útil pero a veces necesita muchos trials para ser informativo
            # fig_slice = plot_slice(study)
            # fig_slice.write_image(self.visualizations_dir / f"{model_name_prefix}_optuna_slice.png")
            self.logger.info(f"Optuna visualizations for {model_name_prefix} saved.")
        except Exception as e:
            self.logger.warning(f"Could not generate/save all Optuna visualizations for {model_name_prefix}: {e}")


    def _evaluate_and_report(self, y_true, y_pred, target_names_list):
        self.logger.info("Evaluating trained classifier performance...")
        accuracy = accuracy_score(y_true, y_pred)
        self.logger.info(f"Final Model Accuracy on Test Set: {accuracy:.4f}")
        
        class_report_str = classification_report(y_true, y_pred, target_names=target_names_list)
        self.logger.info(f"Classification Report:\n{class_report_str}")

        report_dict = classification_report(y_true, y_pred, target_names=target_names_list, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        try:
            report_df.to_csv(self.reports_dir / 'final_classification_report.csv')
            self.logger.info(f"Classification report saved to {self.reports_dir / 'final_classification_report.csv'}")
        except IOError as e:
            self.logger.error(f"Failed to save classification report: {e}")

        self._plot_performance_visuals(y_true, y_pred, target_names_list)

    def _plot_performance_visuals(self, y_true, y_pred, class_labels):
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Absolute Counts)')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')

        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='GnBu', xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
        axes[1].set_title('Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'confusion_matrices_summary.png')
        plt.close(fig)
        self.logger.info(f"Confusion matrices saved to {self.visualizations_dir / 'confusion_matrices_summary.png'}")

        report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
        metrics_for_plot = {k: v for k,v in report_dict.items() if k in class_labels}
        metrics_df = pd.DataFrame(metrics_for_plot).T[['precision', 'recall', 'f1-score']]
        
        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 7))
        metrics_df.plot(kind='bar', ax=ax_metrics, colormap='viridis', width=0.8)
        ax_metrics.set_title('Per-Class Performance Metrics (Precision, Recall, F1-Score)')
        ax_metrics.set_xlabel('Activity Class')
        ax_metrics.set_ylabel('Score')
        ax_metrics.tick_params(axis='x', rotation=45)
        ax_metrics.legend(loc='best')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'per_class_metrics_barchart.png')
        plt.close(fig_metrics)
        self.logger.info(f"Per-class metrics bar chart saved to {self.visualizations_dir / 'per_class_metrics_barchart.png'}")
        
        if hasattr(self.trained_classifier, 'feature_importances_'):
            importances = pd.Series(self.trained_classifier.feature_importances_, index=self.dataset_column_names)
            top_importances = importances.nlargest(25)
            fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
            top_importances.plot(kind='barh', ax=ax_fi, color='teal')
            ax_fi.set_title(f'Top 25 Feature Importances ({type(self.trained_classifier).__name__})')
            ax_fi.invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'top_feature_importances.png')
            plt.close(fig_fi)
            self.logger.info(f"Feature importances plot saved to {self.visualizations_dir / 'top_feature_importances.png'}")


    def _persist_trained_artifact(self):
        if self.trained_classifier is None:
            self.logger.error("No classifier has been trained. Aborting artifact persistence.")
            return

        artifact_package = {
            'classifier_instance': self.trained_classifier,
            'data_scaler_instance': self.data_scaler,
            'target_encoder_instance': self.target_encoder,
            'feature_column_names': self.dataset_column_names,
            'encoded_class_labels': list(self.target_encoder.classes_)
        }
        try:
            with open(self.model_artifact_path, 'wb') as f:
                pickle.dump(artifact_package, f)
            self.logger.info(f"Trained artifact package saved to {self.model_artifact_path}")
        except IOError as e:
            self.logger.error(f"Failed to save trained artifact: {e}")

    def execute_pipeline(self, perform_tuning=True, test_proportion=0.25, optuna_trials=20):
        self.logger.info("Automated Model Pipeline Execution Started.")
        
        features_data, target_data = self._ingest_and_prepare_data()

        X_dev, X_eval, y_dev, y_eval = train_test_split(
            features_data, target_data, test_size=test_proportion, 
            random_state=123, stratify=target_data
        )
        
        X_dev_scaled = self.data_scaler.fit_transform(X_dev)
        X_eval_scaled = self.data_scaler.transform(X_eval)
        
        self.logger.info(f"Data split: Development set size: {X_dev_scaled.shape[0]}, Evaluation set size: {X_eval_scaled.shape[0]}")

        if perform_tuning or not self.config_path.exists():
            self.logger.info("Proceeding with hyperparameter tuning.")
            configurations = self._find_optimal_configurations(X_dev_scaled, y_dev, n_trials_optuna=optuna_trials)
        else:
            self.logger.info(f"Loading existing optimal configurations from {self.config_path}")
            try:
                with open(self.config_path, 'r') as f:
                    configurations = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                self.logger.error(f"Failed to load configurations: {e}. Triggering new tuning process.")
                configurations = self._find_optimal_configurations(X_dev_scaled, y_dev, n_trials_optuna=optuna_trials)

        model_choice = configurations['selected_model_type']
        
        if model_choice == 'random_forest':
            best_hyperparams = configurations['random_forest_config']['params'].copy()
            # Remove Optuna-specific parameters that are not valid for RandomForestClassifier
            best_hyperparams.pop('use_max_depth', None)
            self.trained_classifier = RandomForestClassifier(**best_hyperparams, random_state=42, n_jobs=-1)
        elif model_choice == 'xgboost':
            best_hyperparams = configurations['xgboost_config']['params']
            self.trained_classifier = xgb.XGBClassifier(**best_hyperparams, random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        else:
            self.logger.error(f"Invalid model type in configuration: {model_choice}. Aborting.")
            return

        self.logger.info(f"Training final {model_choice.upper()} model with optimal hyperparameters...")
        try:
            self.logger.info("Entrenando modelo final...")
            self.trained_classifier.fit(X_dev_scaled, y_dev)
            self.logger.info("Entrenamiento finalizado.")
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento final: {e}")
            raise
        
        predictions_on_eval = self.trained_classifier.predict(X_eval_scaled)
        self.logger.info("Predicción sobre el set de evaluación completada.")
        
        self._evaluate_and_report(y_eval, predictions_on_eval, list(self.target_encoder.classes_))
        self._persist_trained_artifact()
        
        self.logger.info("Automated Model Pipeline Execution Finished.")


def run_pose_classification_training():
    pipeline_manager = AutomatedModelPipeline(
        dataset_path='outputRecords_balanceado.csv', # Asegúrate que este sea tu archivo de datos
        output_directory="pose_classification_run_output"
    )
    pipeline_manager.execute_pipeline(perform_tuning=True, optuna_trials=125)

if __name__ == "__main__":
    run_pose_classification_training()