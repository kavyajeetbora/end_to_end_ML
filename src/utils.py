import os
import sys
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle


def save_model(file_path: str, obj: object):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)

        with open(file_path, "wb") as pkl_file:
            dill.dump(obj, pkl_file)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as pkl_handler:
            obj = dill.load(pkl_handler)

        return obj
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():

            model_params = params[model_name]

            logging.info(f"Performing Grid Search CV for model: {model_name}")
            gs = GridSearchCV(estimator=model, param_grid=model_params, cv=3)
            gs.fit(X_train, y_train)

            ## Train the model
            logging.info(f"Grid search completed and best params: {gs.best_params_}")
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Train model prediction
            y_train_pred = model.predict(X_train)

            # Test model prediction
            y_test_pred = model.predict(X_test)

            ## Evaluate the model
            train_score = r2_score(y_pred=y_train_pred, y_true=y_train)
            test_score = r2_score(y_pred=y_test_pred, y_true=y_test)

            report.setdefault(model_name, test_score)

        return report

    except Exception as e:
        raise CustomException(e, sys)
