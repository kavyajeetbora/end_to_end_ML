import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from data_transformation import DataTransformer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Iniated!")

        try:
            logging.info("Reading the data from source")
            df = pd.read_csv(r"notebook\data\stud.csv")

            logging.info("Data is read as dataframe")

            ## Create the artifact folders
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data copied to the artifacts folder")

            logging.info("Performing train test split")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_data.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info(
                "Data splitted into train and test set and exported to artifacts!"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    ## Data Ingestion
    data_injestion_obj = DataIngestion()
    train_data_path, test_data_path = data_injestion_obj.initiate_data_ingestion()

    ## Data Transformation
    transformer = DataTransformer()
    X_train, X_test = transformer.initiate_data_transformation(
        train_data_path, test_data_path
    )

    print(X_train.shape, X_test.shape)
