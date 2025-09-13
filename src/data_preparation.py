# Standard library imports
import logging
import re
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreparation:
    """
    A class used to clean and preprocess hotel booking and sales data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataPreparation with configuration and DataFrame.

        Args:
            config (dict): Configuration dictionary.
            df (pd.DataFrame): Input DataFrame.
        """        
        self.config = config
        self.preprocessor = None # Will be set in build_pipelines_with_resampling
                
        logging.info("Preprocessor initialized successfully.")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the new data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        try:
            # Validate required columns
            required_columns = ['price', 'branch', 'country', 'platform', 'first_time', 'booking_month', 'arrival_month', 'arrival_day', 'checkout_month', 'checkout_day', 'room', 'num_adults', 'num_children']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 'no_show' -> Drop rows where 'no_show' is NaN
            ############################################################################################################################################################################################################################
            df = df.dropna(subset=['no_show'])  # Drop rows where 'no_show' is NaN, so now df has 119,390 rows.

            # 'no_show', 'arrival_day', 'checkout_day', 'num_children' -> Convert to numeric then to int
            ############################################################################################################################################################################################################################
            df[['no_show', 'arrival_day', 'checkout_day', 'num_children']] = df[['no_show', 'arrival_day', 'checkout_day', 'num_children']].apply(pd.to_numeric).astype(int)

            # booking_month -> Convert to categorical with ordered months
            ############################################################################################################################################################################################################################
            month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            df['booking_month_ordered'] = pd.Categorical(df['booking_month'], categories=month_order, ordered=True) # type: ignore

            # Clean 'arrival_month' and convert to ordered categorical
            ############################################################################################################################################################################################################################
            standard_months = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
            df['arrival_month'] = df['arrival_month'].apply(lambda x: x.strip().title() if pd.notna(x) and x.strip().lower() in standard_months else x)
            df['arrival_month_ordered'] = pd.Categorical(df['arrival_month'], categories=month_order, ordered=True) # type: ignore

            # Data Engineering: booking_lead_time
            ############################################################################################################################################################################################################################
            df['booking_month_code'] = df['booking_month_ordered'].cat.codes
            df['arrival_month_code'] = df['arrival_month_ordered'].cat.codes
            month_diff = df['arrival_month_code'] - df['booking_month_code'] + 0.5 # Add 0.5 for bookings made in the same month # +0.5 to ensure same month bookings have lead time of 0.5 months.
            month_diff_adjusted = month_diff + 12 * (month_diff < 0)  # Cross-year adjustment (vectorized)
            df['booking_lead_time'] = month_diff_adjusted.clip(lower=0)

            # Clean 'checkout_month' and convert to ordered categorical
            ############################################################################################################################################################################################################################
            df['checkout_month_ordered'] = pd.Categorical(df['checkout_month'], categories=month_order, ordered=True) # type: ignore

            # Data Engineering: Stay_Duration
            ############################################################################################################################################################################################################################
            df['checkout_month_code'] = df['checkout_month_ordered'].cat.codes
            month_diff = df['checkout_month_code'] - df['arrival_month_code']
            month_diff_adjusted = month_diff + 12 * (month_diff < 0)  # Cross-year adjustment (vectorized)
            day_diff = df['checkout_day'] - df['arrival_day']
            df['Stay_Duration'] = (month_diff_adjusted * 31 + day_diff).clip(lower=0)

            # Drop temporary code columns if desired (optional)
            df = df.drop(['checkout_month_code', 'arrival_month_code'], axis=1)

            # room: There are 21,612 NaN values. I would KNN-impute for these values.
            ############################################################################################################################################################################################################################
            room_mapping = {'Single': 0, 'Queen': 1, 'King': 2, 'President Suite': 3}

            # Apply the mapping to the data_channel column
            df['room_enum'] = df['room'].map(room_mapping)

            # Select features for imputation 
            imputation_features = ['branch', 'booking_month_ordered', 'arrival_month_ordered', 'arrival_day', 'checkout_month_ordered', 'checkout_day', 'country', 'platform', 'room_enum']

            # Create a copy of the dataframe with selected features
            df_impute = df[imputation_features].copy()

            # Encode categorical variables
            df_impute = pd.get_dummies(df_impute, columns=['branch', 'booking_month_ordered', 'arrival_month_ordered', 'checkout_month_ordered', 'country', 'platform'], drop_first=True) # type: ignore

            # Initialize and fit KNN imputer
            imputer = KNNImputer(n_neighbors=5, weights='uniform')  # You can adjust n_neighbors
            df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns) # type: ignore

            # Replace the original column with imputed values
            df['room_enum'] = df_imputed['room_enum']

            # Handle NaN values by filling with a default (e.g., mode or 0) before conversion
            df['room_enum'] = df['room_enum'].fillna(0)  # Use 0 for 'Single' as a fallback

            # Round to the nearest integer and clip to valid categories
            df['room_enum'] = np.round(df['room_enum']).astype(int)
            df['room_enum'] = df['room_enum'].clip(lower=0, upper=3)

            # Map back to original categories
            mapping_room = {0:'Single', 1:'Queen', 2:'King', 3:'President Suite'}
            df['room'] = df['room_enum'].map(mapping_room)

            # price: There are 24,881 null values values. I would KNN-impute for these values.
            ############################################################################################################################################################################################################################
             
            # Extract price information in SGD.
            df['price'] = df['price'].apply(self.extract_SGD_price_info) 

            # Select features for imputation 
            imputation_features = ['branch', 'booking_month_ordered', 'arrival_month_ordered', 'arrival_day', 'checkout_month_ordered', 'checkout_day', 'country', 'platform', 'room_enum', 'num_children', 'price']

            # Create a copy of the dataframe with selected features
            df_impute = df[imputation_features].copy()

            # Encode categorical variables
            df_impute = pd.get_dummies(df_impute, columns=['branch', 'booking_month_ordered', 'arrival_month_ordered', 'checkout_month_ordered', 'country', 'platform'], drop_first=True) # type: ignore

            # Initialize and fit KNN imputer
            imputer = KNNImputer(n_neighbors=5, weights='uniform')  # You can adjust n_neighbors
            df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns) # type: ignore

            # Replace the original column with imputed values
            df['price'] = df_imputed['price']
            median_price = df['price'].median()
            df['price'] = df['price'].fillna(median_price)

            # Feature Engineering: total_cost_of_stay
            ############################################################################################################################################################################################################################
            # Create 'total_cost_of_stay' feature
            df['total_cost_of_stay'] = df['Stay_Duration'] * df['price']

            # 'num_adults' -> Data Cleaning: Convert textual numbers to integers
            ############################################################################################################################################################################################################################
            df['num_adults'] = df['num_adults'].replace({'one': 1, 'two': 2}).astype(int)

            # Feature Engineering: Holiday_or_Biz
            ############################################################################################################################################################################################################################
            # Create 'Holiday_or_Biz' feature
            df['Holiday_or_Biz'] = np.where(
            ((df['num_children'].fillna(0) > 0) |  # Holiday if any children
            (df['num_adults'].fillna(1) > 2)),    # Holiday if more than 2 adults
            'Holiday', 'Biz')

            # Drop unnecessary columns
            df = df.drop(['booking_id', 'booking_month_ordered', 'arrival_month_ordered', 'checkout_month_ordered', 'booking_month_code', 'room_enum'], axis=1)
            print(f"Columns after cleaning: {df.columns.tolist()}")

            logging.info("Data cleaning completed.")
            return df
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise
            
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Splits the DataFrame into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The cleaned DataFrame.

        Returns:
        --------
        Tuple: X_train, y_train, X_val, y_val, X_test, y_test
        """
        logging.info("Starting data splitting.")
        try:
            # Separate features and target variable
            X = df.drop([self.config.get("target_column")], axis=1)
            y = df[self.config.get("target_column")] 
                                     
            # Data split: 80% train, 10% validation, 10% test
            X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=self.config.get("val_test_size"), random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=self.config.get("test_size"), random_state=42, stratify=y_val_test)
                        
            # Ensure all column names are strings
            X_train.columns = [str(col) for col in X_train.columns]
            X_val.columns = [str(col) for col in X_val.columns]
            X_test.columns = [str(col) for col in X_test.columns]
            
            logging.info("Data splitting completed.")
            return  X_train, y_train, X_val, y_val, X_test, y_test
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise


    def build_pipelines_with_resampling(self, df: pd.DataFrame) -> dict:
        """
        Builds a dictionary of Pipelines, each containing preprocessing,
        resampling (SMOTE and RandomUnderSampler), and a classifier.
        """
        # Define feature groups
        numerical_features = self.config.get("numerical_features",{})
        nominal_features = self.config.get("nominal_features",{})
        ordinal_features = self.config.get("ordinal_features",{})
        
        # Validate feature columns
        available_columns = df.columns.tolist()
        for feature_list, name in [
            (numerical_features, "numerical_features"),
            (nominal_features, "nominal_features"),
            (ordinal_features, "ordinal_features")
        ]:
            if not isinstance(feature_list, list):
                raise ValueError(f"{name} must be a list, got {type(feature_list)}: {feature_list}")
            missing_cols = [col for col in feature_list if col not in available_columns]
            if missing_cols:
                raise ValueError(f"Columns in {name} not found in DataFrame: {missing_cols}")

        print(f"Numerical features selected: {numerical_features}")
        print(f"Nominal features selected: {nominal_features}")
        print(f"Ordinal features selected: {ordinal_features}")

        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        # Corrected OrdinalEncoder categories
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(categories=[
                ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], 
                ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                list(range(1, 32)),  # Days 1-31
                ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                list(range(1, 32)),  # Days 1-31
                ['Single', 'Queen', 'King', 'President Suite'],
                [1, 2], # num_adults
                [0, 1, 2, 3] # num_children
            ], handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        # Create a preprocessor using ColumnTransformer with ensured string column names
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('nom', nominal_transformer, nominal_features),
                ('ord', ordinal_transformer, ordinal_features),
            ],
            remainder='drop',
            n_jobs=1,
            verbose_feature_names_out=False
        )
        
        # Define resampling steps for imbalanced classes
        # smote = SMOTE(sampling_strategy={1: 47756}, random_state=42)  # type: ignore # Oversample 'High' # Due to 80% x 119,390 = 95,512; 20% x 119,390 = 23,878; hence to balance, we need to oversample 'High' to 47,756.
        # undersample = RandomUnderSampler(sampling_strategy={0: 47756}, random_state=42)  # type: ignore # Undersample 'Low' # To balance, we need to undersample 'Low' to 47,756.

        # Define the models
        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'LogisticRegression': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(random_state=42, n_jobs=1), 
            'LightGBM': LGBMClassifier(random_state=42, n_jobs=1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0)  # verbose=0 to suppress output
            }

        # Wrap each model in an imblearn.Pipeline
        pipelines = {}
        for name, model in models.items():
            pipelines[name] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                # ('smote', smote),
                # ('undersampler', undersample),
                ('classifier', model)
            ])
        
        return pipelines

    @staticmethod
    def extract_SGD_price_info(price_str: str) -> float: # pyright: ignore[reportReturnType]
        """
        Convert price information which could be in USD or SGD from a string format to a float in SGD.
            Args:
                price_str (str): The price as a string.
            Returns:
                float: The price which is a float in SGD.
        """

        if pd.isna(price_str):
            return None # pyright: ignore[reportReturnType]
                
        # Regular expression to extract currency and price
        # Matches "USD$" or "SGD$" followed by a number (with optional decimal)
        match = re.search(r'(USD\$|SGD\$)\s*([\d\.]+)', price_str)
                
        if match:
            currency, price_value = match.groups()
            price_value = float(price_value)  # Convert matched number to float
                    
            # Convert to SGD based on currency
            if currency == 'USD$':
                return price_value * 1.35  # Assuming 1 USD = 1.35 SGD
            elif currency == 'SGD$':
                return price_value
        else:
            # Handle cases with no valid currency match (e.g., empty or malformed strings)
            number_match = re.search(r'([\d\.]+)', price_str)
            if number_match:
                return float(number_match.group(1))  # Return as SGD if no currency specified
            return None # pyright: ignore[reportReturnType]