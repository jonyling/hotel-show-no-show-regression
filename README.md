# jonyling-RegressionMiniProject
# 'no_show' Prediction - ML model to predict the number of no_shows for hotel bookings to minimize related expenses.

A machine learning project to predict  the number of no_shows for a hotel chain in Singapore, using data preprocessing and model training pipelines. Built with Python, this project leverages libraries like Pandas, Scikit-learn, and YAML for configuration.

Overview
This repository contains the deliverables for the AIAP Foundation Regression Mini Project, addressing the objectives of predicting the number of no_shows on hotel bookings for two hotels of a hotel chain in Singapore, using the dataset provided, and also to evaluate at least 3 suitable models for predicting the number of no_shows. 

Repository Structure

RegressionMiniProject
├── .github/                    # GitHub Actions scripts (provided in template)
├── src/                        # Python modules for ML pipeline
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── model_training.py       # Model training and evaluation
│   └── config.yaml             # Configuration file for pipeline parameters
├── data/                       # Data folder
├── eda.ipynb                   # Jupyter notebook for Task 1 (EDA)
├── requirements.txt            # Python dependencies
├── main.py                     # Main module execute the pipeline
└── README.md                   # This file

## Table of Contents
- [Pipeline Execution Instructions](#PipelineExecutionInstructions)
- [PipelineLogicalFlow](#PipelineLogicalFlow)
- [Key_EDA_Findings_and_Pipeline_Choices](#Key_EDA_Findings_and_Pipeline_Choices)
- [Model_Selection_and_Justification](#Model_Selection_and_Justification)
- [Model_Evaluation](#Model_Evaluation)
- [Configuration](#configuration)
- [MLmodel_results](#MLmodel_results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## PipelineExecutionInstructions
Setup:
- Ensure Python 3.11+ is installed.
- Place mini_project_2_data.csv in the data/ folder (relative path: data/mini_project_2_data.csv).
- Install dependencies: pip install -r requirements.txt.
- Execute the main.py: This runs data_preparation.py to clean data, split data and create the preprocessor. This is followed running model_training.py to train and evaluate for base models, then tuned models, and finally the final model.
- Modify Parameters: Edit src/config.yaml to adjust model hyperparameters, feature selections, or preprocessing steps. Example: Change regressor__alpha for RandomizedSearch. 

## PipelineLogicalFlow
- Data Loading: Load data from data/mini_project_2_data.csv.
- Data Preprocessing:
    - Drop row where 'no_show' is 115536 and all other features are NaN. 
    - Conversion to int32: 'no_show', 'arrival_day', 'checkout_day', 'num_children'
    - Clean 'arrival_month' to standard format. Had entries like noVemBer. 
    - Convert to categorical with ordered months: 'booking_month', 'arrival_month', 'checkout_month'
    - 'num_adults' -> Data Cleaning: Convert textual numbers to integers    
- Feature engineering: 
    - KNN-impute missing values: 'room', 'price'.
    - Data Engineering: 'booking_lead_time', 'Stay_Duration', 'total_cost_of_stay', 'Holiday_or_Biz'
    - Drop features: 'booking_id', 'booking_month_ordered', 'arrival_month_ordered', 'checkout_month_ordered', 'booking_month_code', 'room_enum'
- Split Data:
    - Split data with 80/10/10 proportion for training, validation and test sets.
- Build Pipelines
    - Numerical features =  ['price', 'booking_lead_time', 'Stay_Duration', 'total_cost_of_stay']
    - Ordinal features = ['booking_month', 'arrival_month', 'arrival_day', 'checkout_month', 'checkout_day', 'room', 'num_adults', 'num_children']
    - Nominal features = ['branch', 'country', 'platform', 'first_time', 'Holiday_or_Biz']
- Baseline Models Training: 
    - Train six models: DecisionTree, RandomForest, LogisticRegression, XGBoost, LightGBM, CatBoost. 
- Tuned Models Training: 
    - Train three models: RandomForest, XGBoost and CatBoost, tune hyperparameters with RandomizedSearchCV. 
- Evaluation:
    - Firstly, evaluate models using Accuracy, Macro F1, Weighted F1, Confusion Matrix, and AUC_PR, then secondly, evaluate models with a focus on AUC_PR. It also uses cross_val_score from scikit-learn using AUC_PR for robust validation across folds. 
    - AUC_PR is chosen because the dataset is slightly imbalanced and the task is to predict the minority (no_show=1) class. AUC_PR is the most suitable metric to use in this situation because it summarizes the trade-off between precision and recall, across different probability thresholds, with higher values indicating better performance in identifying the positive class. Other metrics such as accuracy are by nature biased because they can predict the majority class better simply by the proportion that it has. F1-Score (Macro/Weighted) balances precision and recall but is threshold-dependent (requires choosing a single threshold, e.g., 0.5).
    - Final Test Metrics: {RandomForest: AUC_PR = 0.7732876077261251; CatBoost:  AUC_PR = 0.7327303405028012; XGBoost: AUC_PR = 0.7434141312148963}
- Visualization: Below is a simplified flowchart of the pipeline:
    graph TD
        A[Load Data] --> B[Preprocess Data]
        B --> C[Feature Engineering]
        C --> D[Split Data]
        D --> E[Build Pipelines]
        E --> F[Train Baseline Models]
        F --> G[Train Tuned Models]
        G --> H[Evaluation ]

## Key_EDA_Findings_and_Pipeline_Choices
- booking_id 115536 has NaN values in all other columns -> Likely to be data entry error; removed this row. 
- Data Quality: Missing values: 
    - room                           21612
    - price                          24881
- branch: 0(Changi): 79211; 1(Orchard): 40180; roughly the same proportion as show/no_show.
- booking_lead_time: Found to have a statistically significant relationship with no_show, and explains a moderate portion of the variation in no-show.
- country: Found to have a statistically significant relationship with no_show, and explains the highest portion of the variation in no-show.
- Cross-Tabulations for Categorical Interactions: Shows Changi has slightly higher no-show proportion; China gives higher no-show proportion. Also, shows first-timers have a higher no-show proportion across all platforms. 
- Target variable no_shows: Proportion of show/no_show = 63% / 37% 

## Model_Selection_and_Justification
- 6 different models are used in the EDA and the best results, in terms of AUC_PR, of each are:
    DecisionTree                0.6702
    RandomForest                0.7630
    LogisticRegression          0.6254
    XGBoost                     0.7253
    LightGBM                    0.7144
    CatBoost                    0.7285
    
    It is obvious that the top 3 models are RandomForest, CatBoost and LightGBM. 
- It might be due to the following reasons:
    - RandomForest reduces overfitting by averaging predictions across many trees (100–200 in your config). This stabilizes predictions, improving generalization on the validation set. class_weight='balanced' assigns higher weights to the minority class (no-show), improving recall for no-shows, which boosts AUC-PR. It inherently handles high-cardinality categorical features (e.g., country) via one-hot encoding, though it may benefit from fewer categories compared to gradient-boosting methods
    - CatBoost uses symmetric trees and a smaller learning rate (0.01–0.3) to prevent overfitting, especially with your moderately deep trees (depth: [4, 6, 8]). This is more robust than DecisionTree and sometimes XGBoost, which can overfit with deeper trees. It processes categorical features by computing statistics (e.g., target encoding with ordered boosting) during training, reducing the need for one-hot encoding and handling high-cardinality features like country efficiently.
    - LightGBM: The dataset (~95,512 training rows) benefits from LightGBM’s histogram-based splitting, which reduces memory usage and speeds up training compared to XGBoost. This allows LightGBM to handle the 17 features efficiently. Leaf-wise growth (vs. level-wise in XGBoost) builds deeper, more expressive trees, capturing complex patterns.

## Model_Evaluation
- Using the following metrics to evaluate the models:
    - Accuracy 
    - Macro F1
    - Weighted F1
    - Confusion Matrix 
    - AUC_PR 
    - Cross-Validation using AUC_PR. 

    - Ultimately, AUC_PR is used to rank the models. Final Test Metrics: {RandomForest: AUC_PR = 0.7732876077261251; CatBoost:  AUC_PR = 0.7327303405028012; XGBoost: AUC_PR = 0.7434141312148963}

## Configuration
The project uses a config.yaml file for configuration. Key parameters include:
    file_path: Path to the dataset (e.g., mini_project_2_data.csv).
    target_column: The column to predict (e.g., no_show).
    val_test_size: Validate and Test set size (e.g., 0.2).
    param_grid: Hyperparameter grid for the various models (e.g., RandomForest classifier__n_estimators: [100, 200]).
    scoring: AUC_PR.
    numerical_features, nominal_features, ordinal_features: Lists of feature types for preprocessing.
Edit config.yaml to customize the pipeline for your dataset.

## MLmodel_results
- RandomForest: 'auc_pr': 0.7732876077261251, 'accuracy': 0.7658932908953848, 'macro_f1': 0.7370853100387305, 'weighted_f1': 0.7596462160992936, 'confusion_matrix': array([[6548,  969], [1826, 2596]], dtype=int64), 
'cv_auc_pr': 0.7622340998014426
- CatBoost: 'auc_pr': 0.7327303405028012, 'accuracy': 0.7580199346678951, 'macro_f1': 0.7248007614736571, 'weighted_f1': 0.7495870056964553, 'confusion_matrix': array([[6599,  918], [1971, 2451]], dtype=int64), 
'cv_auc_pr': 0.723421069635889 
- XGBoost: 'auc_pr': 0.7434141312148963, 'accuracy': 0.7588575257559259, 'macro_f1': 0.7293520915063254, 'weighted_f1': 0.7525178243836754, 'confusion_matrix': array([[6501, 1016], [1863, 2559]], dtype=int64), 
'cv_auc_pr': 0.7340099700098783

## Contributing
We welcome contributions! To contribute:
    Fork the repository.
    Create a new branch (git checkout -b feature/your-feature-name).
    Commit your changes (git commit -m 'Add your feature').
    Push to the branch (git push origin feature/your-feature-name).
    Open a Pull Request.
Please follow our Code of Conduct (CODE_OF_CONDUCT.md) and ensure code adheres to PEP 8 style guidelines.

## License
This project is licensed under the MIT License (LICENSE).

## Contact 
For questions or feedback, reach out to:
    Email: jonyling@hotmail.com
    GitHub: jonyling
    X: @JonyLing1

This README is concise yet informative, tailored to your project's structure. Adjust the repository URL, contact details, and dataset path as needed. Let me know if you'd like to expand any section!
