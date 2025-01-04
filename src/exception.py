"""
CUSTOM ERROR HANDLING

Here we will define a custom exception class that will be used in the entire project
Whenever we define try catch block in any code of this project, we will raise this Exception class
and will print the error message with this custom message we have define here
"""

import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    This function returns a custom error message to the user for debugging purpose
    """
    _, _, exec_tb = error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename
    line_no = exec_tb.tb_lineno

    error_message = f"Error occured in python script {file_name} at line number {line_no} | Error Message: {str(error)}"

    return error_message


## Create a custom exception for our project
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message


if __name__ == "__main__":

    logging.info("Logging has started")
    try:
        a = 1 / 0
    except Exception as e:
        errorObject = CustomException(e, sys)
        logging.info(errorObject.__str__())
        raise errorObject
