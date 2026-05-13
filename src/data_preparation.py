"""Data cleaning, feature engineering, and preprocessing for no-show prediction."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
ROOM_ORDER = ["Single", "Queen", "King", "President Suite"]
STANDARD_MONTHS = {month.lower() for month in MONTH_ORDER}


class DataPreparation:
    """Clean raw hotel booking data and build model-ready preprocessing."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.target_column = config.get("target_column", "no_show")
        self.random_state = int(config.get("random_state", 42))

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the EDA notebook's cleaning and feature engineering choices."""
        cleaned = df.copy()
        self._validate_raw_columns(cleaned)

        cleaned[self.target_column] = cleaned[self.target_column].replace("", np.nan)
        cleaned = cleaned.dropna(subset=[self.target_column]).copy()
        cleaned[self.target_column] = pd.to_numeric(cleaned[self.target_column]).astype(int)

        cleaned["arrival_day"] = pd.to_numeric(cleaned["arrival_day"], errors="coerce")
        cleaned["checkout_day"] = pd.to_numeric(cleaned["checkout_day"], errors="coerce").abs()
        cleaned["num_children"] = pd.to_numeric(cleaned["num_children"], errors="coerce").fillna(0).astype(int)
        cleaned["num_adults"] = cleaned["num_adults"].replace({"one": 1, "two": 2})
        cleaned["num_adults"] = pd.to_numeric(cleaned["num_adults"], errors="coerce").fillna(1).astype(int)

        for column in ["booking_month", "arrival_month", "checkout_month"]:
            cleaned[column] = cleaned[column].apply(self._standardize_month)

        cleaned["arrival_day"] = cleaned["arrival_day"].fillna(cleaned["arrival_day"].median())
        cleaned["checkout_day"] = cleaned["checkout_day"].fillna(cleaned["checkout_day"].median())
        cleaned["arrival_day"] = cleaned["arrival_day"].clip(lower=1, upper=31).astype(int)
        cleaned["checkout_day"] = cleaned["checkout_day"].clip(lower=1, upper=31).astype(int)

        cleaned["booking_lead_time"] = self._month_difference(
            cleaned["booking_month"], cleaned["arrival_month"], same_month_offset=0.5
        )
        cleaned["Stay_Duration"] = self._stay_duration(cleaned)
        cleaned["Stay_Duration_logged"] = np.log1p(cleaned["Stay_Duration"].astype(int))

        room_mapping = {room: index for index, room in enumerate(ROOM_ORDER)}
        room_enum = cleaned["room"].map(room_mapping)
        room_mode = room_enum.mode(dropna=True)
        fallback_room = int(room_mode.iloc[0]) if not room_mode.empty else 0
        cleaned["room"] = room_enum.fillna(fallback_room).round().clip(0, 3).astype(int).map(dict(enumerate(ROOM_ORDER)))

        cleaned["is_usd"] = cleaned["price"].apply(lambda value: int(isinstance(value, str) and value.strip().startswith("USD$")))
        cleaned["price"] = cleaned["price"].apply(self.extract_sgd_price)
        cleaned["price"] = cleaned["price"].fillna(cleaned["price"].median())

        cleaned["total_cost_of_stay"] = cleaned["Stay_Duration"] * cleaned["price"]
        cleaned["Holiday_or_Biz"] = np.where(
            (cleaned["num_children"].fillna(0) > 0) | (cleaned["num_adults"].fillna(1) > 2),
            "Holiday",
            "Biz",
        )
        cleaned["total_guests"] = cleaned["num_adults"] + cleaned["num_children"]
        cleaned["has_children"] = (cleaned["num_children"] > 0).astype(int)
        cleaned["price_per_person"] = cleaned["price"] / (cleaned["total_guests"] + 1e-6)
        cleaned["price_per_night"] = cleaned["price"] / (cleaned["Stay_Duration"] + 1e-6)
        cleaned["is_solo_traveler"] = (cleaned["total_guests"] == 1).astype(int)
        cleaned["is_long_stay"] = (cleaned["Stay_Duration"] > 7).astype(int)
        cleaned["is_last_minute_booking"] = (cleaned["booking_month"] == cleaned["arrival_month"]).astype(int)
        cleaned["lead_time_category"] = pd.cut(
            cleaned["booking_lead_time"],
            bins=[-1, 7, 30, 90, 180, np.inf],
            labels=["Last Minute", "Short Term", "Medium Term", "Long Term", "Very Long Term"],
        )
        cleaned["is_international"] = (cleaned["country"] != "Singapore").astype(int)
        cleaned["is_direct_booking"] = (cleaned["platform"] == "Website").astype(int)
        cleaned["is_peak_season"] = cleaned["arrival_month"].isin(["June", "July", "November", "December"]).astype(int)

        model_columns = self.model_features + [self.target_column]
        missing_model_columns = [column for column in model_columns if column not in cleaned.columns]
        if missing_model_columns:
            raise ValueError(f"Cleaned data is missing configured model columns: {missing_model_columns}")

        cleaned = cleaned[model_columns].copy()
        LOGGER.info("Cleaned data shape: %s", cleaned.shape)
        return cleaned

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create the notebook's stratified 80/20 train-test split."""
        X = df[self.model_features]
        y = df[self.target_column].astype(int)

        return train_test_split(
            X,
            y,
            test_size=float(self.config.get("test_size", 0.2)),
            random_state=self.random_state,
            stratify=y,
        )

    def build_preprocessor(self) -> ColumnTransformer:
        """Build the ColumnTransformer used by every classifier pipeline."""
        self._validate_feature_groups()

        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        ordinal_transformer = Pipeline(
            steps=[
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[MONTH_ORDER, MONTH_ORDER, MONTH_ORDER, ROOM_ORDER],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("nom", nominal_transformer, self.nominal_features),
                ("ord", ordinal_transformer, self.ordinal_features),
            ],
            remainder="drop",
            n_jobs=1,
            verbose_feature_names_out=False,
        )

    @property
    def numerical_features(self) -> list[str]:
        return list(self.config.get("numerical_features", []))

    @property
    def nominal_features(self) -> list[str]:
        return list(self.config.get("nominal_features", []))

    @property
    def ordinal_features(self) -> list[str]:
        return list(self.config.get("ordinal_features", []))

    @property
    def model_features(self) -> list[str]:
        return self.numerical_features + self.nominal_features + self.ordinal_features

    @staticmethod
    def extract_sgd_price(price_value: Any) -> float:
        """Convert SGD/USD price strings into SGD floats."""
        if pd.isna(price_value):
            return np.nan
        if isinstance(price_value, (int, float, np.integer, np.floating)):
            return float(price_value)

        price_str = str(price_value).strip()
        match = re.search(r"(USD\$|SGD\$)\s*([\d]+(?:\.\d+)?)", price_str)
        if match:
            currency, parsed_price = match.groups()
            price = float(parsed_price)
            return price * 1.35 if currency == "USD$" else price

        number_match = re.search(r"([\d]+(?:\.\d+)?)", price_str)
        return float(number_match.group(1)) if number_match else np.nan

    @staticmethod
    def _standardize_month(value: Any) -> Any:
        if pd.isna(value):
            return value
        month = str(value).strip()
        return month.title() if month.lower() in STANDARD_MONTHS else month

    @staticmethod
    def _month_codes(months: pd.Series) -> pd.Series:
        return pd.Categorical(months, categories=MONTH_ORDER, ordered=True).codes

    def _month_difference(self, start_month: pd.Series, end_month: pd.Series, same_month_offset: float = 0.0) -> pd.Series:
        start_codes = self._month_codes(start_month)
        end_codes = self._month_codes(end_month)
        month_diff = end_codes - start_codes + same_month_offset
        return pd.Series(month_diff + 12 * (month_diff < 0), index=start_month.index).clip(lower=0)

    def _stay_duration(self, df: pd.DataFrame) -> pd.Series:
        month_diff = self._month_difference(df["arrival_month"], df["checkout_month"])
        day_diff = df["checkout_day"] - df["arrival_day"]
        return (month_diff * 30 + day_diff).clip(lower=0)

    def _validate_raw_columns(self, df: pd.DataFrame) -> None:
        required_columns = {
            "booking_id",
            self.target_column,
            "branch",
            "booking_month",
            "arrival_month",
            "arrival_day",
            "checkout_month",
            "checkout_day",
            "country",
            "first_time",
            "room",
            "price",
            "platform",
            "num_adults",
            "num_children",
        }
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(f"Raw data is missing required columns: {missing_columns}")

    def _validate_feature_groups(self) -> None:
        for name, features in {
            "numerical_features": self.numerical_features,
            "nominal_features": self.nominal_features,
            "ordinal_features": self.ordinal_features,
        }.items():
            if not features:
                raise ValueError(f"{name} must contain at least one feature")
            if not all(isinstance(feature, str) for feature in features):
                raise ValueError(f"{name} must contain only strings")
