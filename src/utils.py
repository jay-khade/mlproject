# ----------------------------------------------- importing libraries -------------------------------------

import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging


# ----------------------------------------- utility functions ----------------------------------------------

def save_object(file_path, obj):
    """
    Function to save objects/pkl files
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        CustomException(e, sys)