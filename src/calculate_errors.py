from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd


def symmetric_mean_absolute_error(actual, predicted):
    return 100 / len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


def calculate_error(err_func, y_true, y_pred, error_list, ts_list, ts_idx, error_name, level_list):
    err = err_func(y_true, y_pred)
    error_list.append([ts_list[ts_idx], error_name, err, level_list[ts_idx]])


def calculate_errors_per_fc(data, fc_type, actual_test, file_name, half_horizon_case):
    forecast_type = fc_type.split('_')
    fc_type_ml = forecast_type[0]
    errors = []
    if fc_type == 'base':
        df_forecasts = pd.read_csv(f"forecasts/new_data_samples/{file_name}_forecasts.csv", index_col=1).iloc[:, 1:]
    elif fc_type_ml == 'case1' or fc_type_ml == 'case2':
        df_forecasts = pd.read_csv(f"results/expanding_window_results/ml_fc/{file_name}_{fc_type}.csv", index_col=0)
    else:
        if fc_type == 'mintsample' and (data == 'prison' or data == 'wikipedia'):
            return
        else:
            df_forecasts = pd.read_csv(f"results/expanding_window_results/benchmarks/{file_name}_{fc_type}.csv",
                                       index_col=0)

    # iterate through each time series in hierarchy
    ts_names = actual_test.index.values
    levels = actual_test['Level'].values
    for ts in range(len(actual_test)):
        if half_horizon_case:
            idx = half_horizon[data]
            actual_test_ts = actual_test.iloc[ts: ts + 1, 1:idx + 1].values[0]
            # fc
            ts_fc = df_forecasts.iloc[ts: ts + 1, 0:idx].values[0]
        else:
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
    if len(forecast_type) > 3:
        fc_type = f'{forecast_type[0]}_{forecast_type[1]}_{forecast_type[2]}_no_skip'
    if half_horizon_case:
        file_name = f'{file_name}_short_horizon'
    errors.to_csv(f'results/expanding_window_results/errors/{file_name}_{fc_type}.csv')


def run_errors(data, model, half_horizon_case):
    print(data, model, half_horizon_case)

    FC_TYPE = ['base', 'bottomup', 'ols', 'wls', 'mintsample', 'mintshrink', 'erm',
               'case1_lambda_1',
               'case1_lambda_[0.01, 0.09]',
               'case1_lambda_[0.1, 0.9]',
               'case1_lambda_[1, 4]',
               'case1_lambda_[0.01, 5]',
               'case2_lambda_1',
               'case2_lambda_[0.01, 0.09]',
               'case2_lambda_[0.1, 0.9]',
               'case2_lambda_[1, 4]',
               'case2_lambda_[0.01, 5]',
               'case1_lambda_1_lambda_1_no_skip',
               'case1_lambda_[0.01, 0.09]_lambda_[0.01, 0.09]_no_skip',
               'case1_lambda_[0.1, 0.9]_lambda_[0.1, 0.9]_no_skip',
               'case1_lambda_[1, 4]_lambda_[1, 4]_no_skip',
               'case1_lambda_[0.01, 5]_lambda_[0.01, 5]_no_skip',
               'case2_lambda_1_lambda_1_no_skip',
               'case2_lambda_[0.01, 0.09]_lambda_[0.01, 0.09]_no_skip',
               'case2_lambda_[0.1, 0.9]_lambda_[0.1, 0.9]_no_skip',
               'case2_lambda_[1, 4]_lambda_[1, 4]_no_skip',
               'case2_lambda_[0.01, 5]_lambda_[0.01, 5]_no_skip']

    for fc_type in FC_TYPE:
        # file_name = f'{data}_{model}'

        # full dataset one sample
        # actual_test = pd.read_csv(f"input_data/{data}_test.csv", index_col=1)
        # calculate_errors_per_fc(fc_type, actual_test, file_name)

        samples = dataset_samples[data]
        for sample in range(0, samples):
            actual_test = pd.read_csv(f"input_data/new_data_samples/{data}_{sample}_test.csv", index_col=1)
            file_name = f'{data}_{sample}_{model}'
            calculate_errors_per_fc(data, fc_type, actual_test, file_name, half_horizon_case)


if __name__ == '__main__':
    half_horizon = {'prison': 4, 'tourism': 6, 'wikipedia': 3, 'labour': 4}
    dataset_samples = {'prison': 3, 'tourism': 10, 'wikipedia': 10, 'labour': 5}

    datasets = ['prison', 'labour', 'tourism', 'wikipedia']
    half_horizon_cases = [True, False]
    models = ['arima', 'ets']

    for data in datasets:
        for model in models:
            for half_horizon_val in half_horizon_cases:
                run_errors(data, model, half_horizon_val)
