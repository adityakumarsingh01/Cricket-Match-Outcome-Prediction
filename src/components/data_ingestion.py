import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join(
            "data",
            "raw",
            "IPL_B2B_Dataset.csv"
        )

        self.processed_dir = os.path.join(
            "data",
            "processed"
        )

        self.train_data_path = os.path.join(
            self.processed_dir,
            "train.csv"
        )

        self.test_data_path = os.path.join(
            self.processed_dir,
            "test.csv"
        )

    def initiate_data_ingestion(self):
        """
        Reads the raw cricket dataset, performs basic data cleaning,
        splits the data into training and testing sets, and saves them
        inside data/processed/.
        """

        try:
            logging.info("Starting cricket data ingestion")

            # Create processed directory if it does not exist
            os.makedirs(
                self.processed_dir,
                exist_ok=True
            )

            # Read raw dataset
            df = pd.read_csv(
                self.raw_data_path,
                low_memory=False
            )

            logging.info(
                f"Raw dataset loaded successfully: {df.shape}"
            )

            # Remove unnecessary columns
            columns_to_drop = [
                'Index',
                'Unnamed: 11',
                'Unnamed: 12'
            ]

            existing_columns_to_drop = [
                col for col in columns_to_drop
                if col in df.columns
            ]

            df = df.drop(
                columns=existing_columns_to_drop
            )

            logging.info(
                "Removed unnecessary identifier/empty columns"
            )

            # Convert required run rate to numeric
            df['req_run_rate'] = pd.to_numeric(
                df['req_run_rate'],
                errors='coerce'
            )

            # Replace infinite values with NaN
            df = df.replace(
                [float('inf'), float('-inf')],
                pd.NA
            )

            # Remove invalid/terminal match states
            df = df[
                (df['balls_left'] > 0) &
                (df['runs_left'] > 0)
            ].copy()

            logging.info(
                "Removed invalid/terminal match states"
            )

            # Handle missing categorical values
            df['city'] = df['city'].fillna(
                'Unknown'
            )

            # Handle missing numerical values
            df['req_run_rate'] = pd.to_numeric(
                df['req_run_rate'],
                errors='coerce'
            )

            df['req_run_rate'] = df['req_run_rate'].fillna(
                df['req_run_rate'].median()
            )

            logging.info(
                "Missing values handled successfully"
            )

            # Remove duplicate rows
            before_duplicates = len(df)

            df = df.drop_duplicates().reset_index(
                drop=True
            )

            after_duplicates = len(df)

            logging.info(
                f"Removed {before_duplicates - after_duplicates} "
                f"duplicate rows"
            )

            # Separate features and target
            X = df.drop(
                columns=['result']
            )

            y = df['result']

            # Stratified train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            # Reconstruct train and test datasets
            train_df = X_train.copy()
            train_df['result'] = y_train

            test_df = X_test.copy()
            test_df['result'] = y_test

            # Save processed datasets
            train_df.to_csv(
                self.train_data_path,
                index=False
            )

            test_df.to_csv(
                self.test_data_path,
                index=False
            )

            logging.info(
                f"Training dataset saved: "
                f"{self.train_data_path}"
            )

            logging.info(
                f"Testing dataset saved: "
                f"{self.test_data_path}"
            )

            logging.info(
                f"Training shape: {train_df.shape}"
            )

            logging.info(
                f"Testing shape: {test_df.shape}"
            )

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            logging.error(
                "Exception occurred during data ingestion"
            )

            raise CustomException(
                e,
                sys
            )