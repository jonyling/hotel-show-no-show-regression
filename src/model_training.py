# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTraining:
    """
    A class used to train and evaluate machine learning models on predicting no_shows on hotel bookings.

    Attributes:
    -----------
    cconfig : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    pipelines : Dict[str, ImbPipeline]
        Dictionary of preprocessing and model pipelines for each algorithm.
    """

    def __init__(self, config: Dict[str, Any], pipelines: Dict[str, Pipeline]):
        """
        Initialize the ModelTraining class with configuration.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        pipelines: Preprocessor pipelines created in DataPreparation class.
        """
        # Validate config
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary")
        self.config = config
        
        # Validate and set pipelines
        if not isinstance(pipelines, dict):
            raise ValueError("pipelines must be a dictionary")
        if not all(isinstance(pipe, Pipeline) for pipe in pipelines.values()):
            raise ValueError("All pipeline values must be Pipeline objects")
        self.pipelines = pipelines
        
        if not isinstance(config, dict):
            raise ValueError("pipelines must be of the correct type")
            

    def train_and_evaluate_baseline_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        pipelines: Dict[str, Pipeline]
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Train and evaluate baseline models using provided pipelines.
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_test: Test features
            y_test: Test target values
            pipelines: Dictionary of sklearn pipelines to evaluate
            
        Returns:
            Tuple containing:
            - Dictionary of trained pipeline objects
            - Dictionary of evaluation metrics for each model
        """
        baseline_metrics = {}
        baseline_models = {}
        
        for name, pipeline in pipelines.items():
            try:
                print(f"Fitting {name}...")
                # Fit the entire pipeline
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)
                y_pred_proba = pipeline.predict_proba(X_val)[:, 1]  # Probability for no-show (1)

                # Calculate AUC-PR
                precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
                auc_pr = auc(recall, precision)

                # Store metrics
                baseline_metrics[name] = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'macro_f1': f1_score(y_val, y_pred, average='macro'),
                    'weighted_f1': f1_score(y_val, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_val, y_pred), 
                    'auc_pr': auc_pr,
                }

                # Perform cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=1)
                baseline_metrics[name]['cv_auc_pr'] = cv_scores.mean()  # Use consistent key

                # Store the fitted pipeline
                baseline_models[name] = pipeline  # Add this line

                # Save the trained pipeline
                os.makedirs('models', exist_ok=True)
                joblib.dump(pipeline, f'models/{name}.pkl')
                
                print(f"\nShow-No_show Report ({name}):")
                print(classification_report(y_val, y_pred, target_names=['Show', 'No_show']))

            except Exception as e:
                logging.error(f"Error fitting {name}: {e}")
                continue

        # Print metrics
        for name, metrics in baseline_metrics.items():
            print(f"\nModel: {name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f}")
            print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"CV AUC-PR: {metrics['cv_auc_pr']:.4f}")  # Fix key and label
            print("Confusion Matrix:")
            print(metrics['confusion_matrix'])
            print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        
        return baseline_models, baseline_metrics

        

    def train_and_evaluate_tuned_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        top_3_pipelines: Dict[str, Pipeline]
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Train and evaluate tuned models using provided pipelines.
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_test: Validation features
            y_test: Validation target values
            top_3_pipelines: Dictionary of top 3 pipelines (by AUC-PR) to tune and evaluate
            
        Returns:
            Tuple containing:
            - Dictionary of tuned pipeline objects
            - Dictionary of evaluation metrics for each model
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        param_grid = self.config.get("param_grid", {})
        scoring = self.config.get("scoring", "average_precision")

        logging.info(f"Loaded param_grid: {param_grid}")

        try:
            for name, pipeline in top_3_pipelines.items():
                try:
                    print(f"Fitting {name} with hyperparameter tuning...")
                    model_param_grid = param_grid.get(name, {})
                    if not model_param_grid:
                        logging.warning(f"No param_grid for {name}; using default parameters")
                        best_pipeline = pipeline
                        best_pipeline.fit(X_train, y_train)
                        cv_score = cross_val_score(
                            best_pipeline, X_train, y_train, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                            scoring=scoring, n_jobs=-1
                        ).mean()
                    else:
                        # Ensure all param_grid values are lists or arrays
                        for param, values in model_param_grid.items():
                            if not isinstance(values, (list, np.ndarray)):
                                logging.warning(f"Invalid param_grid for {name}, {param}: {values}. Wrapping in list.")
                                model_param_grid[param] = [values]
                        # Wrap param_grid in a list to satisfy RandomizedSearchCV
                        search = RandomizedSearchCV(
                            pipeline,
                            param_distributions=[model_param_grid],  # Wrap in list
                            n_iter=10,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            scoring=scoring,
                            n_jobs=-1,
                            random_state=42
                        )
                        search.fit(X_train, y_train)
                        best_pipeline = search.best_estimator_
                        cv_score = search.best_score_
                        print(f"Best parameters for {name}: {search.best_params_}")

                    # Evaluate on validation set
                    y_pred = best_pipeline.predict(X_test) # type: ignore[misc]
                    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1] # type: ignore[misc]

                    # Calculate AUC-PR
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    auc_pr = auc(recall, precision)

                    tuned_metrics[name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
                        'auc_pr': auc_pr,
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'cv_auc_pr': cv_score
                    }

                    tuned_models[name] = best_pipeline
                    os.makedirs('models', exist_ok=True)
                    joblib.dump(best_pipeline, f'models/tuned_{name}.pkl')
                    print(f"\nShow-No_show Report ({name}):")
                    print(classification_report(y_test, y_pred, target_names=['Show', 'No_show']))

                except Exception as e:
                    logging.error(f"Error tuning {name}: {e}")
                    continue

            # Print metrics
            for name, metrics in tuned_metrics.items():
                print(f"\nModel: {name}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Macro F1: {metrics['macro_f1']:.4f}")
                print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
                print(f"CV AUC-PR: {metrics['cv_auc_pr']:.4f}")
                print("Confusion Matrix:")
                print(metrics['confusion_matrix'])
                print(f"AUC-PR: {metrics['auc_pr']:.4f}")
            
            logging.info("Hyperparameter tuning completed.")
            return tuned_models, tuned_metrics

        except Exception as e:
            logging.error(f"Error tuning models: {e}")
            raise
