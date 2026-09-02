import sys
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException


class DataTransformation:
    def __init__(self):
        self.preprocessor = None

        self.numerical_features = [
            'runs_left',
            'balls_left',
            'wickets',
            'target_runs',
            'cur_run_rate',
            'req_run_rate'
        ]

        self.categorical_features = [
            'batting_team',
            'bowling_team',
            'city'
        ]

    def get_data_transformer_object(self):
        """
        Creates the preprocessing pipeline for cricket match data.
        """

        try:
            logging.info("Creating cricket data transformation pipeline")

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(strategy='median')
                    ),
                    (
                        'scaler',
                        StandardScaler()
                    )
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(strategy='most_frequent')
                    ),
                    (
                        'one_hot_encoder',
                        OneHotEncoder(
                            handle_unknown='ignore',
                            sparse_output=False
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'numerical',
                        numerical_pipeline,
                        self.numerical_features
                    ),
                    (
                        'categorical',
                        categorical_pipeline,
                        self.categorical_features
                    )
                ]
            )

            logging.info("Data transformation pipeline created successfully")

            return preprocessor

        except Exception as e:
            logging.error(
                "Error while creating data transformation pipeline"
            )
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test datasets, applies preprocessing,
        and returns transformed arrays.
        """

        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                f"Train data shape: {train_df.shape}"
            )

            logging.info(
                f"Test data shape: {test_df.shape}"
            )

            target_column = 'result'

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Replace infinite values with NaN
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)

            # Remove invalid/terminal match states
            train_valid = (
                (X_train['balls_left'] > 0) &
                (X_train['runs_left'] > 0)
            )

            test_valid = (
                (X_test['balls_left'] > 0) &
                (X_test['runs_left'] > 0)
            )

            X_train = X_train[train_valid].copy()
            y_train = y_train.loc[X_train.index]

            X_test = X_test[test_valid].copy()
            y_test = y_test.loc[X_test.index]

            logging.info(
                "Removed invalid/terminal match states"
            )

            self.preprocessor = self.get_data_transformer_object()

            X_train_transformed = self.preprocessor.fit_transform(
                X_train
            )

            X_test_transformed = self.preprocessor.transform(
                X_test
            )

            logging.info(
                "Data transformation completed successfully"
            )

            logging.info(
                f"Transformed train shape: "
                f"{X_train_transformed.shape}"
            )

            logging.info(
                f"Transformed test shape: "
                f"{X_test_transformed.shape}"
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.preprocessor
            )

        except Exception as e:
            logging.error(
                "Exception occurred during data transformation"
            )
            raise CustomException(e, sys)