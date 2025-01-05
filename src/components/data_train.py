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

            ## Model parameters
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            ## Evaluate the models
            logging.info("Evaluating the models")
            report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
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
