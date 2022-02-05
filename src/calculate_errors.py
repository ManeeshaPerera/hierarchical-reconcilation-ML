from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd


def symmetric_mean_absolute_error(actual, predicted):
    return 100 / len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


def calculate_error(err_func, y_true, y_pred, y_adjusted, error_list, ts_list, ts_idx, error_name, level_list):
    err = err_func(y_true, y_pred)
    err_adjusted = err_func(y_true, y_adjusted)
    error_list.append([ts_list[ts_idx], 'original', error_name, err, level_list[ts_idx]])
    error_list.append([ts_list[ts_idx], 'err_adjusted', error_name, err_adjusted, level_list[ts_idx]])


if __name__ == '__main__':
    data = 'prison'
    model = 'ets'
    file_name = f'{data}_{model}'
    errors = []
    actual_train = pd.read_csv(f"input_data/{data}_actual.csv", index_col=1)
    actual_test = pd.read_csv(f"input_data/{data}_test.csv", index_col=1)
    df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv", index_col=1)
    # calculating the errors
    adjusted_fc = pd.read_csv(f"results/{file_name}_adjusted_forecasts.csv", index_col=0)
    # iterate through each time series in hierarchy
    ts_names = actual_train.index.values
    levels = actual_train['Level'].values
    for ts in range(len(actual_test)):
        actual_train_ts = actual_train.iloc[ts: ts + 1, 1:].values[0]
        actual_test_ts = actual_test.iloc[ts: ts + 1, 1:].values[0]
        # original fc
        ts_fc_original = df_forecasts.iloc[ts: ts + 1, 1:].values[0]
        # adjusted fc
        ts_fc_adjusted = adjusted_fc.iloc[ts: ts + 1, :].values[0]

        calculate_error(mean_squared_error, actual_test_ts, ts_fc_original, ts_fc_adjusted, errors, ts_names, ts, 'MSE',
                        levels)
        calculate_error(mean_absolute_error, actual_test_ts, ts_fc_original, ts_fc_adjusted, errors, ts_names, ts,
                        'MAE', levels)
        calculate_error(symmetric_mean_absolute_error, actual_test_ts, ts_fc_original, ts_fc_adjusted, errors, ts_names,
                        ts, 'SMAPE', levels)

    errors = pd.DataFrame(errors)
    errors.columns = ['ts_name', 'run', 'forecast_type', 'error_metric', 'error', 'level']
    errors.to_csv(f'results/{file_name}_errors.csv')
