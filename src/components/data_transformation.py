from src.logger import logging
from src.exception import CustomException
from src.utils import save_model

import os
import sys

# from data_ingestion import DataIngestion

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor_obj.pkl")


class DataTransformer:
    def __init__(self):
        self.processor_obj_file_path = DataTransformationConfig()

    def get_data_transformer(self):

        try:

            ## First list down the categorical columns and numerical columns separately
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_features = ["reading_score", "writing_score"]

            logging.info(f"Categorical features: {cat_features}")
            logging.info(f"Numerical features: {numerical_features}")

            ## Create the data cleaining pipeline for numerical
            numerical_pipeline = Pipeline(
                steps=[
                    ("Impute", SimpleImputer(strategy="median")),
                    ("StandardScaler", StandardScaler()),
                ]
            )
            logging.info(
                "Numerical columns missing value imputation and standard scaling completed"
            )

            ## Create the data clearning and transformation pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                ]
            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical pipeline", numerical_pipeline, numerical_features),
                    ("categorical_pipeline", cat_pipeline, cat_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            ## First read the train and test dataset

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("The train and test data are read!")

            ## Create the preprocessor object
            logging.info("Obtaining preprocessing object")

            preprocessor = self.get_data_transformer()
            logging.info("Preprocessor object created")

            ## Save the preprocessor object
            save_model(
                file_path=self.processor_obj_file_path.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            logging.info("Successfully exported the preprocessor object")

            ## Set the target and dependent variable
            target_column = "math_score"

            X_train = train_data.drop([target_column], axis=1)
            X_test = test_data.drop([target_column], axis=1)

            logging.info(
                f"Input feature for train with {X_train.shape} and test with {X_test.shape} are created"
            )

            ## Now Transform the data for training
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logging.info(
                "Successfully Transformed the train and test input features. Ready for training"
            )

            return (X_train, X_test)

        except Exception as e:
            raise CustomException(e, sys)
