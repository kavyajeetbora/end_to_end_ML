import os
import sys
from src.logger import logging
from src.exception import CustomException
import dill


def save_model(file_path: str, obj: object):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)

        with open(file_path, "wb") as pkl_file:
            dill.dump(obj, pkl_file)

    except Exception as e:
        raise CustomException(e, sys)
