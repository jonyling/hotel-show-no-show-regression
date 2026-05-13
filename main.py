"""Run the hotel booking no-show classification pipeline."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hotel no-show classification models.")
    parser.add_argument("--config", default=PROJECT_DIR / "src" / "config.yaml", type=Path)
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV after baseline training.")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU parameters for XGBoost, LightGBM, and CatBoost.")
    parser.add_argument(
        "--tune-top-n",
        type=int,
        default=None,
        help="Tune only the top N baseline models by AUC-PR. Defaults to all tuned models in config.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    start_time = time.time()

    config = load_config(args.config)
    config["use_gpu"] = bool(args.gpu or config.get("use_gpu", False))
    config["models_dir"] = str(PROJECT_DIR / config.get("models_dir", "models"))
    config["results_dir"] = str(PROJECT_DIR / config.get("results_dir", "results"))

    data_path = PROJECT_DIR / config.get("file_path", "data/mini_project_2_data.csv")
    LOGGER.info("Loading data from %s", data_path)
    raw_df = pd.read_csv(data_path)

    data_preparation = DataPreparation(config)
    cleaned_df = data_preparation.clean_data(raw_df)
    X_train, X_test, y_train, y_test = data_preparation.split_data(cleaned_df)
    LOGGER.info("Train shape: %s, test shape: %s", X_train.shape, X_test.shape)
    LOGGER.info("Target distribution: %s", y_train.value_counts(normalize=True).round(3).to_dict())

    model_training = ModelTraining(config, data_preparation.build_preprocessor())
    _, baseline_metrics = model_training.train_baseline_models(X_train, y_train, X_test, y_test)

    print("\nBaseline metrics ranked by AUC-PR")
    print(baseline_metrics[["model", "accuracy", "macro_f1", "weighted_f1", "cv_auc_pr", "auc_pr"]])

    if args.tune:
        tuned_model_names = config.get("tuned_models")
        if args.tune_top_n:
            tuned_model_names = baseline_metrics.head(args.tune_top_n)["model"].tolist()

        _, tuned_metrics = model_training.tune_models(
            X_train,
            y_train,
            X_test,
            y_test,
            model_names=tuned_model_names,
        )
        print("\nTuned metrics ranked by AUC-PR")
        print(tuned_metrics[["model", "accuracy", "macro_f1", "weighted_f1", "cv_auc_pr", "auc_pr"]])

    elapsed_minutes = (time.time() - start_time) / 60
    LOGGER.info("Pipeline finished in %.2f minutes", elapsed_minutes)


if __name__ == "__main__":
    main()
