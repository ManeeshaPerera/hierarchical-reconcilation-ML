from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
import numpy as np
import pandas as pd


def symmetric_mean_absolute_error(actual, predicted):
    return 100 / len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


if __name__ == '__main__':
    data = 'prison'
    model = 'ets'
    file_name = f'{data}_{model}'
    actual_train = pd.read_csv(f"input_data/{data}_actual.csv", index_col=1)
    actual_test = pd.read_csv(f"input_data/{data}_test.csv", index_col=1)
    df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv", index_col=1)
    # calculating the errors for each run
    errors = []
    for run in range(5):
        adjusted_fc = pd.read_csv(f"results/{file_name}_adjusted_forecasts_{run}.csv", index_col=0)
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

            for error in ['MASE', 'SMAPE']:
                if error == 'MASE':
                    original_error = mean_absolute_scaled_error(y_true=actual_test_ts, y_pred=ts_fc_original,
                                                                y_train=actual_train_ts)
                    adjusted_error = mean_absolute_scaled_error(y_true=actual_test_ts, y_pred=ts_fc_adjusted,
                                                                y_train=actual_train_ts)
                    errors.append([ts_names[ts], run + 1, 'original', 'MASE', original_error, levels[ts]])
                    errors.append([ts_names[ts], run + 1, 'adjusted', 'MASE', adjusted_error, levels[ts]])

                else:
                    original_error = symmetric_mean_absolute_error(actual_test_ts, ts_fc_original)
                    adjusted_error = symmetric_mean_absolute_error(actual_test_ts, ts_fc_adjusted)
                    errors.append([ts_names[ts], run + 1, 'original', 'SMAPE', original_error, levels[ts]])
                    errors.append([ts_names[ts], run + 1, 'adjusted', 'SMAPE', adjusted_error, levels[ts]])

    errors = pd.DataFrame(errors)
    errors.columns = ['ts_name', 'run', 'forecast_type', 'error_metric', 'error', 'level']
    errors.to_csv(f'results/{file_name}_errors.csv')
