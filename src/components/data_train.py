from src.logger import logging
from src.exception import CustomException
from src.utils import save_model, evaluate_model
import os
import sys

## skearn
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trainedModelFilePath = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_file_path_obj = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info(
                "Extracting input and target variables from train and test arrays"
            )
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            ## Models dictionary
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            ## Evaluate the models
            logging.info("Evaluating the models")
            report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            logging.info("Evaluation of models completed")

            logging.info("Choosing the best model")
            sorted_report = sorted(report.items(), key=lambda x: x[1], reverse=True)
            best_model_name, best_model_score = sorted_report[0]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model is {best_model_name} with a evaluation R2 score of {best_model_score:.2f}"
            )
            best_model = models[best_model_name]

            ## Make Prediction on test data set
            y_test_pred = best_model.predict(X_test)
            best_model_r2_score = r2_score(y_true=y_test, y_pred=y_test_pred)

            save_model(
                file_path=self.model_file_path_obj.trainedModelFilePath, obj=best_model
            )
            return best_model_r2_score

        except Exception as e:
            raise CustomException(e, sys)
