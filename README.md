# Hotel Booking No-Show Classification

This project predicts whether a hotel booking will become a no-show. The pipeline was rewritten from `eda.ipynb` so the production code follows the same cleaning, feature engineering, model selection, and evaluation choices used in the notebook.

## Repository Structure

```text
jonyling-ClassificationMiniProject/
├── data/
│   └── mini_project_2_data.csv
├── src/
│   ├── config.yaml
│   ├── data_preparation.py
│   └── model_training.py
├── eda.ipynb
├── main.py
├── requirements.txt
└── README.md
```

Generated model files are written to `models/`, and metrics are written to `results/`. Both folders are ignored by git.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run baseline training:

```bash
python main.py
```

Run baseline training plus hyperparameter tuning:

```bash
python main.py --tune
```

Tune only the top three baseline models by AUC-PR:

```bash
python main.py --tune --tune-top-n 3
```

Enable GPU parameters for XGBoost, LightGBM, and CatBoost when your environment supports them:

```bash
python main.py --gpu --tune
```

## Pipeline Flow

1. Load `data/mini_project_2_data.csv`.
2. Drop the row with missing `no_show` and convert the target to integer classes.
3. Clean date-like fields, month casing, day values, guest counts, room, and price.
4. Engineer notebook-selected features:
   `booking_lead_time`, `Stay_Duration`, `Stay_Duration_logged`, `is_usd`, `total_cost_of_stay`, `Holiday_or_Biz`, `total_guests`, `has_children`, `price_per_person`, `price_per_night`, `is_solo_traveler`, `is_long_stay`, `is_last_minute_booking`, `lead_time_category`, `is_international`, `is_direct_booking`, and `is_peak_season`.
5. Keep the final model feature set from the notebook:
   17 numerical features, 5 nominal features, and 4 ordinal features.
6. Split data using a stratified 80/20 train-test split.
7. Train Decision Tree, Random Forest, Logistic Regression, XGBoost, LightGBM, and CatBoost pipelines.
8. Evaluate using accuracy, macro F1, weighted F1, confusion matrix, classification report, cross-validated average precision, and AUC-PR.

## EDA Choices Reflected in Code

- The target is imbalanced at roughly 63% show and 37% no-show, so AUC-PR/average precision is the primary ranking metric.
- `country` had the strongest categorical association with no-show among the explored fields.
- `booking_lead_time`, `branch`, `arrival_month`, `checkout_month`, `first_time`, and guest-count features showed weaker but useful signal.
- SMOTE and random undersampling were tested in the notebook but not used because they worsened AUC-PR.
- Price strings are converted to SGD; USD prices use the notebook conversion of `1 USD = 1.35 SGD`.
- Missing room values use the encoded room mode, and missing price values use the median price, matching the final notebook cells.

## Notebook Results

The notebook's tuned run ranked the models by AUC-PR as:

| Model | AUC-PR | CV Avg Precision | Accuracy |
| --- | ---: | ---: | ---: |
| RandomForest | 0.7642 | 0.7603 | 0.7601 |
| LightGBM | 0.7520 | 0.7511 | 0.7611 |
| CatBoost | 0.7330 | 0.7315 | 0.7540 |
| XGBoost | 0.7306 | 0.7251 | 0.7538 |
| DecisionTree | 0.6811 | 0.6673 | 0.7298 |
| LogisticRegression | 0.6217 | 0.6208 | 0.7047 |

## Configuration

Edit `src/config.yaml` to change:

- input path and output folders
- random seed, split size, CV folds, and scoring metric
- numerical, nominal, and ordinal feature lists
- model hyperparameter search spaces
- whether GPU parameters are enabled by default
