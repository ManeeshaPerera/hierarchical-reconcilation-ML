import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope
from src.construct_heirarchy import ConstructHierarchy


class MLReconcile:
    def __init__(self, seed_value, actual_data, fitted_data, forecasts, number_of_levels):
        self.hierarchy = None
        self.actual_data = actual_data
        self.fitted_data = fitted_data
        self.forecasts = forecasts
        self.number_of_levels = number_of_levels
        self._run_initialization(seed_value)
        self.actual_transpose = None
        self.fitted_transpose = None
        self.forecast_transpose = None

    def _run_initialization(self, seed_value):
        tf.random.set_seed(seed_value)  # set seed for tensorflow
        self.actual_transpose = self._transpose_data(self.actual_data)
        self.fitted_transpose = self._transpose_data(self.fitted_data)
        self.forecast_transpose = self._transpose_data(self.forecasts)
        self.hierarchy = ConstructHierarchy(self.fitted_transpose,
                                            self.number_of_levels)  # construct time series hierarchy

    def _transpose_data(self, dataframe):
        dataframe_transpose = dataframe.iloc[:, 1:]
        dataframe_transpose.columns = [col_idx if col_idx != 0 else "" for col_idx in
                                       range(len(dataframe_transpose.columns))]
        dataframe_transpose = dataframe_transpose.set_index("").transpose()
        return dataframe_transpose
