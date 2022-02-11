from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd


def symmetric_mean_absolute_error(actual, predicted):
    return 100 / len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


def calculate_error(err_func, y_true, y_pred, error_list, ts_list, ts_idx, error_name, level_list):
    err = err_func(y_true, y_pred)
    error_list.append([ts_list[ts_idx], error_name, err, level_list[ts_idx]])


if __name__ == '__main__':
    data = 'tourism'
    model = 'ets'
    FC_TYPE = ['base', 'adjusted', 'hf_val_loss']
    file_name = f'{data}_{model}'
    actual_test = pd.read_csv(f"input_data/{data}_test.csv", index_col=1)
    for fc_type in FC_TYPE:
        errors = []
        if fc_type == 'base':
            df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv", index_col=1).iloc[:, 1:]
        elif fc_type == 'adjusted':
            df_forecasts = pd.read_csv(f"results/{file_name}_adjusted_forecasts.csv", index_col=0)
        elif fc_type == 'hf_val_loss':
            df_forecasts = pd.read_csv(f"results/{file_name}_adjusted_forecasts_{fc_type}.csv", index_col=0)

        # iterate through each time series in hierarchy
        ts_names = actual_test.index.values
        levels = actual_test['Level'].values
        for ts in range(len(actual_test)):
            actual_test_ts = actual_test.iloc[ts: ts + 1, 1:].values[0]
            # fc
            ts_fc = df_forecasts.iloc[ts: ts + 1, :].values[0]

            calculate_error(mean_squared_error, actual_test_ts, ts_fc, errors, ts_names, ts, 'MSE',
                            levels)
            calculate_error(mean_absolute_error, actual_test_ts, ts_fc, errors, ts_names, ts,
                            'MAE', levels)
            calculate_error(symmetric_mean_absolute_error, actual_test_ts, ts_fc, errors, ts_names,
                            ts, 'SMAPE', levels)

        errors = pd.DataFrame(errors)
        errors.columns = ['ts_name', 'error_metric', 'error', 'level']
        errors.to_csv(f'results/errors/{file_name}_{fc_type}_errors.csv')
