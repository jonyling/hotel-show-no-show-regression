"""Model construction, training, tuning, evaluation, and persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)


class ModelTraining:
    """Train and evaluate the classifiers selected in the EDA notebook."""

    def __init__(self, config: dict[str, Any], preprocessor: Any):
        self.config = config
        self.preprocessor = preprocessor
        self.random_state = int(config.get("random_state", 42))
        self.models_dir = Path(config.get("models_dir", "models"))
        self.results_dir = Path(config.get("results_dir", "results"))
        self.cv_folds = int(config.get("cv_folds", 5))
        self.scoring = config.get("scoring", "average_precision")
        self.n_iter = int(config.get("n_iter", 10))
        self.use_gpu = bool(config.get("use_gpu", False))

    def build_pipelines(self) -> dict[str, Pipeline]:
        """Create one preprocessing + classifier pipeline per model."""
        models = {
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(random_state=self.random_state, class_weight="balanced"),
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "XGBoost": XGBClassifier(random_state=self.random_state, n_jobs=1, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(random_state=self.random_state, n_jobs=1, verbose=-1),
            "CatBoost": CatBoostClassifier(random_state=self.random_state, verbose=0),
        }

        if self.use_gpu:
            self._enable_gpu(models)

        return {
            name: Pipeline(
                steps=[
                    ("preprocessor", clone(self.preprocessor)),
                    ("classifier", model),
                ]
            )
            for name, model in models.items()
        }

    def train_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[dict[str, Pipeline], pd.DataFrame]:
        """Fit default models and evaluate them on the held-out test set."""
        pipelines = self.build_pipelines()
        fitted_models: dict[str, Pipeline] = {}
        metric_rows: list[dict[str, Any]] = []

        for name, pipeline in pipelines.items():
            LOGGER.info("Fitting baseline %s", name)
            pipeline.fit(X_train, y_train)
            metrics = self.evaluate_model(name, pipeline, X_test, y_test)
            metrics["cv_auc_pr"] = self.cross_validate(pipeline, X_train, y_train)
            fitted_models[name] = pipeline
            metric_rows.append(metrics)
            self.save_model(pipeline, f"{name}.pkl")
            self.print_report(name, metrics)

        metrics_df = self.metrics_to_frame(metric_rows)
        self.save_metrics(metrics_df, "baseline_metrics.csv")
        return fitted_models, metrics_df

    def tune_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_names: list[str] | None = None,
    ) -> tuple[dict[str, Pipeline], pd.DataFrame]:
        """Tune configured models with RandomizedSearchCV and evaluate them."""
        pipelines = self.build_pipelines()
        param_grid = self.config.get("param_grid", {})
        if model_names is None:
            model_names = list(pipelines)

        tuned_models: dict[str, Pipeline] = {}
        metric_rows: list[dict[str, Any]] = []
        cv = self.cv_splitter

        for name in model_names:
            if name not in pipelines:
                LOGGER.warning("Skipping unknown model %s", name)
                continue

            pipeline = pipelines[name]
            model_params = param_grid.get(name, {})
            LOGGER.info("Tuning %s", name)

            if model_params:
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=model_params,
                    n_iter=min(self.n_iter, self._parameter_space_size(model_params)),
                    cv=cv,
                    scoring=self.scoring,
                    n_jobs=1,
                    random_state=self.random_state,
                )
                search.fit(X_train, y_train)
                best_pipeline = search.best_estimator_
                cv_score = search.best_score_
                best_params = search.best_params_
            else:
                best_pipeline = pipeline.fit(X_train, y_train)
                cv_score = self.cross_validate(best_pipeline, X_train, y_train)
                best_params = {}

            metrics = self.evaluate_model(name, best_pipeline, X_test, y_test)
            metrics["cv_auc_pr"] = cv_score
            metrics["best_params"] = best_params
            tuned_models[name] = best_pipeline
            metric_rows.append(metrics)
            self.save_model(best_pipeline, f"tuned_{name}.pkl")
            self.print_report(name, metrics, tuned=True)

        metrics_df = self.metrics_to_frame(metric_rows)
        self.save_metrics(metrics_df, "tuned_metrics.csv")
        return tuned_models, metrics_df

    def evaluate_model(self, name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """Calculate threshold and ranking metrics for one fitted model."""
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

        return {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "auc_pr": auc(recall, precision),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["Show", "No_show"],
                zero_division=0,
            ),
        }

    def cross_validate(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=self.cv_splitter,
            scoring=self.scoring,
            n_jobs=1,
        )
        return float(scores.mean())

    @property
    def cv_splitter(self) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    @staticmethod
    def metrics_to_frame(metric_rows: list[dict[str, Any]]) -> pd.DataFrame:
        metrics_df = pd.DataFrame(metric_rows)
        if metrics_df.empty:
            return metrics_df
        return metrics_df.sort_values("auc_pr", ascending=False).reset_index(drop=True)

    def save_model(self, pipeline: Pipeline, filename: str) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.models_dir / filename)

    def save_metrics(self, metrics_df: pd.DataFrame, filename: str) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(self.results_dir / filename, index=False)

    @staticmethod
    def print_report(name: str, metrics: dict[str, Any], tuned: bool = False) -> None:
        label = "Tuned" if tuned else "Baseline"
        print(f"\n{label} {name}")
        print(metrics["classification_report"])
        print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"CV AUC-PR: {metrics['cv_auc_pr']:.4f}")
        print(f"Confusion matrix: {metrics['confusion_matrix']}")

    @staticmethod
    def _parameter_space_size(param_grid: dict[str, list[Any]]) -> int:
        size = 1
        for values in param_grid.values():
            size *= len(values)
        return max(size, 1)

    @staticmethod
    def _enable_gpu(models: dict[str, Any]) -> None:
        models["XGBoost"].set_params(device="cuda")
        models["LightGBM"].set_params(device="gpu")
        models["CatBoost"].set_params(task_type="GPU", devices="0")
