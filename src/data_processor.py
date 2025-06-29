import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.data = None
        self.X = None
        self.y = None
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.preprocessor = None
        self.feature_names = None
        self.original_feature_names = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully from {self.data_path}. Shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {self.data_path}")
            return False
        except pd.errors.EmptyDataError:
            logging.error(f"Error: Data file at {self.data_path} is empty.")
            return False
        except Exception as e:
            logging.error(f"Error loading data: {e}", exc_info=True)
            return False

    def _apply_capping(self, df, numeric_features, lower_bound_percentile, upper_bound_percentile):
        for col in numeric_features:
            lower_bound = df[col].quantile(lower_bound_percentile)
            upper_bound = df[col].quantile(upper_bound_percentile)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            logging.info(f"Capped column '{col}' at {lower_bound_percentile*100}% ({lower_bound:.2f}) and {upper_bound_percentile*100}% ({upper_bound:.2f}).")
        return df

    def preprocess_data(self, test_size=0.2, val_size=0.25, random_state=42, use_smote=True,
                        apply_capping=True, lower_bound_percentile=0.01, upper_bound_percentile=0.99):
        if self.data is None:
            logging.error("No data loaded. Call load_data() first.")
            return False

        if self.target_column not in self.data.columns:
            logging.error(f"Target column '{self.target_column}' not found in data.")
            return False

        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        self.original_feature_names = self.X.columns.to_numpy()

        if self.y.dtype == 'object':
            logging.info("Converting target column to numeric (0 and 1).")
            unique_targets = self.y.unique()
            if len(unique_targets) == 2:
                self.y = self.y.map({unique_targets[0]: 0, unique_targets[1]: 1})
            else:
                logging.warning(f"Target column has more than 2 unique object values ({unique_targets}). ")
        elif not pd.api.types.is_numeric_dtype(self.y):
             logging.warning(f"Target column '{self.target_column}' is not numeric. Attempting to coerce.")
             self.y = pd.to_numeric(self.y, errors='coerce')
             if self.y.isnull().any():
                 logging.error("Target column contains non-numeric values that cannot be coerced to numbers.")
                 return False

        numeric_features = self.X.select_dtypes(include=np.number).columns
        categorical_features = self.X.select_dtypes(include='object').columns

        if self.X[numeric_features].isnull().sum().sum() > 0:
            logging.warning("Missing values detected in numeric features. Filling with mean.")
            for col in numeric_features:
                if self.X[col].isnull().any():
                    self.X[col] = self.X[col].fillna(self.X[col].mean())

        if self.X[categorical_features].isnull().sum().sum() > 0:
            logging.warning("Missing values detected in categorical features. Filling with mode.")
            for col in categorical_features:
                if self.X[col].isnull().any():
                    self.X[col] = self.X[col].fillna(self.X[col].mode()[0])

        if apply_capping and len(numeric_features) > 0:
            self.X = self._apply_capping(self.X, numeric_features, lower_bound_percentile, upper_bound_percentile)
            logging.info("Outlier capping applied to numeric features.")
        elif apply_capping and len(numeric_features) == 0:
            logging.warning("Capping requested but no numeric features found to apply it to.")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        logging.info(f"Initial split: Train+Val shape {X_train_val.shape}, Test shape {self.X_test.shape}")

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )
        logging.info(f"Further split: Train shape {self.X_train.shape}, Validation shape {self.X_val.shape}")

        self.X_train = self.preprocessor.fit_transform(self.X_train)

        self.X_val = self.preprocessor.transform(self.X_val)
        self.X_test = self.preprocessor.transform(self.X_test)
        logging.info("Features scaled and One-Hot Encoded using ColumnTransformer.")

        self.feature_names = self.preprocessor.get_feature_names_out()
        logging.info(f"Updated feature names. Total features: {len(self.feature_names)}")


        if use_smote:
            class_counts = pd.Series(self.y_train).value_counts()

            min_samples = class_counts.min() if not class_counts.empty else 0

            if len(class_counts) < 2:
                logging.warning("SMOTE skipped: Only one class found in training data. Cannot apply SMOTE.")
            elif min_samples < 2:
                logging.warning(f"SMOTE skipped: Minority class has too few samples ({min_samples}). "
                                "Need at least 2 samples to apply SMOTE with default k_neighbors=1.")
            else:
                logging.info(f"Original training class distribution: {class_counts.to_dict()}")

                smote_k_neighbors = min(5, min_samples - 1)
                if smote_k_neighbors < 1:
                    logging.warning(f"Calculated k_neighbors for SMOTE is {smote_k_neighbors}. SMOTE requires k_neighbors >= 1. Skipping SMOTE.")
                else:
                    smote = SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)
                    self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                    logging.info(f"SMOTE applied with k_neighbors={smote_k_neighbors}. New training class distribution: {pd.Series(self.y_train).value_counts().to_dict()}")

        logging.info("Data preprocessing complete.")
        return True

    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def get_preprocessor(self):
        return self.preprocessor

    def get_feature_names(self):
        return self.feature_names

    def get_original_feature_names(self):
        return self.original_feature_names