# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
import time
from sklearn.utils._testing import ignore_warnings
from pathlib import Path 

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@ignore_warnings(category=Warning) # type: ignore[misc]
def main():

    start_time = time.time()
    print("Starting full pipeline...")

    # Configuration file path
    config_path = "./src/config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load CSV file into a DataFrame
    main_dir = Path(__file__).parent
    data_path = main_dir / "./data/mini_project_2_data.csv"
    try:
        df = pd.read_csv(data_path)
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"Datafile cannot be loaded: {str(e)}")

    # Initialize and run data preparation
    data_prep = DataPreparation(config)  # type: ignore[misc]

    # Clean data and engineeer features
    cleaned_df = data_prep.clean_data(df)  

    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = data_prep.split_data(cleaned_df)
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_train columns: {X_train.columns.tolist()}")

    # Create preprocessors and build pipelines with resampling
    pipelines = data_prep.build_pipelines_with_resampling(cleaned_df) # type: ignore[misc]

    # Initialize and run data preparation
    model_training = ModelTraining(config, pipelines)  # type: ignore[misc]

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (model_training.train_and_evaluate_baseline_models(X_train, y_train, X_val, y_val, pipelines)) # type: ignore[misc]

    # Select top 3 baseline models and feed them for tuning.
    df_metrics = pd.DataFrame.from_dict(baseline_metrics, orient='index') # Convert baseline_metrics to DataFrame (rows: models, columns: metrics)
    df_auc_pr = df_metrics[['auc_pr']].sort_values('auc_pr', ascending=False) # Select only AUC-PR column and sort descending
    top_3_models = df_auc_pr.head(3) # Get top 3
    print("Top 3 Models by AUC-PR:")
    print(top_3_models)
    top_3_names = top_3_models.index.tolist() # Extract top 3 model names (from index)
    top_3_pipelines = {name: baseline_models[name] for name in top_3_names} # Extract top 3 pipelines (dict with only top 3 entries)
    print(f"\nTop 3 Pipelines Keys: {list(top_3_pipelines.keys())}")
   
    # Pass top_3_pipelines, train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(X_train, y_train, X_test, y_test, top_3_pipelines)  

    # Log the results
    logging.info("Baseline Metrics:")
    logging.info(baseline_metrics)
    logging.info("Tuned Metrics:") 
    logging.info(tuned_metrics)
    logging.info("Tuned Models:") 
    logging.info(tuned_models)

if __name__ == "__main__":
    main()

