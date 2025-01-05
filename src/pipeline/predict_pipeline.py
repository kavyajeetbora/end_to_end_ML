import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object


class PredictPipeline:
    """
    This class is responsible to create the prediction pipeline:
    1. Convert the input user data to CustomData format
    2. Then use this CustomData format to predict the results
    3. Return the result to the user

    """

    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading the model and the data preprocessor")
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor_obj.pkl"

            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Successfully loaded the preprocessor")

            model = load_object(file_path=model_path)

            logging.info("Successfully loaded the model")

            ## Now transform the data using preprocessor
            logging.info(f"{features}")
            X = preprocessor.transform(features)
            logging.info("User input transformed successfully")

            ## Now predict the results using the transformed data
            y_pred = model.predict(X)
            logging.info("Prediction completed based on the user input")

            return y_pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethinicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethinicity = race_ethinicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
        """
        This function converts the user input to a dataframe to be used as input for prediction

        """
        input_data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethinicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }

        df = pd.DataFrame(input_data)
        return df
