import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException


def load_raw_data(path: str) -> pd.DataFrame:
    try:
        logging.info("Starting data ingestion for IPL dataset")

        df = pd.read_csv(path)

        logging.info(
            f"Loaded {df.shape[0]} rows and {df.shape[1]} columns"
        )

        return df

    except Exception as e:
        logging.error("Error occurred during IPL data ingestion")

        raise CustomException(e, sys)
