import logging
import os
from datetime import datetime

import sys
from exception import CustomException

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(logs_path, exist_ok=True)  ## Even if folder exists, append the files

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


"""Testing the logger file and CustomException"""
# if __name__ == "__main__":

#     logging.info("Logging has started")
#     try:
#         a = 1 / 0
#     except Exception as e:
#         errorObject = CustomException(e, sys)
#         logging.info(errorObject.__str__())
#         raise errorObject
