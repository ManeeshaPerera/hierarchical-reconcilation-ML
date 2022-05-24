import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from darts import TimeSeries
from darts.models import RNNModel, TCNModel
from darts.utils.likelihood_models import GaussianLikelihood

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


def convert_ts(data_frame, level):
    data_frame = data_frame.loc[data_frame['Level'] == level].iloc[:, 1:].transpose()
    ts = data_frame.columns
    print(data_frame)
    print(ts)
    # data_frame.columns = ['time', 'value']
    # return TimeSeries.from_dataframe(
    #     data_frame, "time", "value"
    # )


train = pd.read_csv('input_data/new_data_samples/labour_0_actual.csv')
test = pd.read_csv('input_data/new_data_samples/labour_0_test.csv')

convert_ts(train, 2)

# my_model = RNNModel(
#     model="LSTM",
#     hidden_dim=20,
#     dropout=0,
#     batch_size=16,
#     n_epochs=50,
#     optimizer_kwargs={"lr": 1e-3},
#     random_state=0,
#     # training_length=50,
#     input_chunk_length=4,
#     output_chunk_length=1,
# )
# #
# # my_model.fit(target_train, verbose=True)
# #
# # pred = my_model.predict(12)
# # print(pred)
#
# print(my_model.historical_forecasts(target_train))
