"""Smoke-test project imports."""

import catboost
import imblearn
import joblib
import lightgbm
import matplotlib
import numpy
import pandas
import scipy
import seaborn
import sklearn
import threadpoolctl
import xgboost
import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

print("All dependencies and local modules imported successfully")
