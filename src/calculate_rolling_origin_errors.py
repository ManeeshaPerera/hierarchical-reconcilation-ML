import sys

from sklearn.metrics import mean_squared_error
import pandas as pd


def calculate_error(err_func, y_true, y_pred, error_list, ts_list, ts_idx, error_name, level_list):
    err = err_func(y_true, y_pred)
    error_list.append([ts_list[ts_idx], err, level_list[ts_idx]])


def calculate_errors_per_fc(data, fc_type, actual_test, model, iteration):
    errors = []
    if fc_type == 'base':
        df_forecasts = pd.read_csv(f"rolling_window_experiments/{data}/{model}_forecasts_{iteration}.csv",
                                   index_col=1).iloc[:, 1:]

    elif fc_type == 'mintsample' and (data == 'prison' or data == 'wikipedia'):
        return
    else:
        df_forecasts = pd.read_csv(f"rolling_window_experiments/hts/{data}/{model}_{fc_type}_{iteration}.csv",
                                   index_col=0)

    # iterate through each time series in hierarchy
    print(fc_type)
    ts_names = actual_test.index.values
    levels = actual_test['Level'].values
    for ts in range(len(actual_test)):
        actual_test_ts = actual_test.iloc[ts: ts + 1, 1:].values[0]
        # fc
        ts_fc = df_forecasts.iloc[ts: ts + 1, :].values[0]
        calculate_error(mean_squared_error, actual_test_ts, ts_fc, errors, ts_names, ts, 'MSE',
                        levels)

    all_errors = pd.DataFrame(errors)
    all_errors.columns = ['ts_name', 'error', 'level']

    # get level wise error
    errors = all_errors.groupby(by=['level']).mean().reset_index().round(2)

    # level wise mean
    errors.index = LEVELS[data]
    overall_error = pd.DataFrame(all_errors[['error']].mean()).transpose()
    overall_error.index = ['Overall']
    errors = errors.append(overall_error).round(2)
    return errors


def run_errors(data, model, errors_per_fc_type):
    samples = ROLLING_WINDOWS[data]
    for fc_type in FC_TYPE:
        sample_errors = []
        for sample in range(1, samples + 1):
            actual_test = pd.read_csv(f"rolling_window_experiments/{data}/test_{sample}.csv", index_col=1)
            sample_error = calculate_errors_per_fc(data, fc_type, actual_test, model, sample)
            sample_errors.append(sample_error)
        all_samples = pd.concat(sample_errors)
        # this will give the mean across all samples for a given method
        mean_error = all_samples.groupby(all_samples.index).mean().reindex(
            sample_errors[0].index.values)
        mean_error.to_csv(f"rolling_window_experiments/results/{data}/{model}_{fc_type}.csv")
        errors_per_fc_type.append(mean_error)


if __name__ == '__main__':
    ROLLING_WINDOWS = {'prison': 24,
                       'tourism': 120,
                       'labour': 60,
                       'wikipedia': 70}

    datasets = ['prison', 'labour', 'tourism', 'wikipedia']
    models = ['arima', 'ets']
    LEVELS = {'tourism': ['Australia', 'States', 'Regions'],
              'prison': ['Australia', 'State', 'Gender', 'Legal', 'Indigenous'],
              'labour': ['Total Employees', 'Main Occupation', 'Employment Status', 'Gender'],
              'wikipedia': ['Total', 'Access', 'Agent', 'Language', 'Purpose', 'Article']}

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
               'case2_lambda_[0.01, 5]']
    data = datasets[int(sys.argv[1])]

    for model in models:
        # # one step ahead horizon
        errors_per_fc_type = []  # 0 index corresponds to base errors
        percentages = []
        run_errors(data, model, errors_per_fc_type)
        # get percentage improvement over base forecasts
        for idx_method in range(1, len(errors_per_fc_type)):
            percentage_improvement = ((errors_per_fc_type[0] - errors_per_fc_type[idx_method]) / errors_per_fc_type[
                0]) * 100
            percentage_improvement = percentage_improvement.iloc[:, 1:]
            percentage_improvement.columns = [FC_TYPE[idx_method]]
            percentages.append(percentage_improvement)
        percentages = pd.concat(percentages, axis=1)
        percentages.sort_values(by='Overall', axis=1, ascending=False).to_csv(
            f'rolling_window_experiments/results/{data}/{model}_error_percentages.csv')
