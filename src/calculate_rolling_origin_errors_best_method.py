import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os


def calculate_error(err_func, y_true, y_pred, error_list, ts_list, ts_idx, level_list):
    err = err_func(y_true, y_pred)
    error_list.append([ts_list[ts_idx], err, level_list[ts_idx]])


def calculate_errors_per_fc(data, fc_type, actual_test, model, iteration, error_name):
    errors = []
    if fc_type == 'base':
        df_forecasts = pd.read_csv(f"rolling_window_experiments/{data}/{model}_forecasts_{iteration}.csv",
                                   index_col=1).iloc[:, 1:]
    elif 'case' in fc_type:
        # transformed values are here
        if (data == 'prison' or data == 'wikipedia') and model == 'arima':
            df_forecasts = pd.read_csv(
                f"rolling_window_experiments_transformed/hts/{data}/{experiment_number}/{model}_{fc_type}_{iteration}.csv",
                index_col=0)
        else:
            df_forecasts = pd.read_csv(
                f"rolling_window_experiments/hts/{data}/{experiment_number}/{model}_{fc_type}_{iteration}.csv",
                index_col=0)

    else:
        df_forecasts = pd.read_csv(
            f"rolling_window_experiments/hts/{data}/{data}_new/{model}_{fc_type}_{iteration}.csv",
            index_col=0)

    # iterate through each time series in hierarchy
    ts_names = actual_test.index.values
    levels = actual_test['Level'].values
    for ts in range(len(actual_test)):
        actual_test_ts = actual_test.iloc[ts: ts + 1, 1:].values[0]
        # fc
        ts_fc = df_forecasts.iloc[ts: ts + 1, :].values[0]
        if error_name == 'MSE':
            calculate_error(mean_squared_error, actual_test_ts, ts_fc, errors, ts_names, ts,
                            levels)
        else:
            calculate_error(mean_absolute_error, actual_test_ts, ts_fc, errors, ts_names, ts,
                            levels)

    all_errors = pd.DataFrame(errors)
    all_errors.columns = ['ts_name', 'error', 'level']

    ts_wise_errors = all_errors.iloc[:, 0:2]
    ts_wise_errors = ts_wise_errors.set_index('ts_name')

    # get level wise error
    errors = all_errors.groupby(by=['level']).mean().reset_index().round(2)

    # level wise mean
    errors.index = LEVELS[data]
    overall_error = pd.DataFrame(all_errors[['error']].mean()).transpose()
    overall_error.index = ['Overall']
    errors = errors.append(overall_error).round(2)
    return {'errors': errors, 'ts_wise_errors': ts_wise_errors}


def run_errors(data, model, errors_per_fc_type, error_name):
    samples = ROLLING_WINDOWS[data]
    for fc_type in FC_TYPE:
        if fc_type == 'mintsample' and (data == 'prison' or data == 'wikipedia'):
            continue
        else:
            sample_errors_overall = []
            for sample in range(1, samples + 1):
                actual_test = pd.read_csv(f"rolling_window_experiments/{data}/test_{sample}.csv",
                                          index_col=1)
                error_dic = calculate_errors_per_fc(data, fc_type, actual_test, model, sample, error_name)
                sample_error = error_dic['errors']
                sample_errors_overall.append(sample_error)
            sample_wise_error = pd.concat(sample_errors_overall, axis=1).drop(columns='level')
            sample_wise_error.columns = [f'sample {i}' for i in range(1, samples + 1)]
            sample_wise_error = sample_wise_error.transpose()[['Overall']].values
            if fc_type != 'base' or 'case' not in fc_type:
                errors_per_fc_type[fc_type] = sample_wise_error
            if 'case' in fc_type:
                ml_errors_per_fc_type[fc_type] = sample_wise_error
                ml_method_errors.append(sample_wise_error.mean())  # ml method overall mean error
                ml_mean[fc_type] = sample_wise_error.mean()


if __name__ == '__main__':
    ROLLING_WINDOWS = {'prison': 24,
                       'tourism': 120,
                       'labour': 60,
                       'wikipedia': 70}

    datasets = ['prison', 'labour', 'tourism', 'wikipedia']
    models = ['arima', 'ets', 'deepAR'', waveNet']
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
               'case2_lambda_[0.01, 5]'
               ]

    FC_TYPE_prison_wiki = ['base', 'bottomup', 'ols', 'wls', 'mintshrink', 'erm',
                           'case1_lambda_1',
                           'case1_lambda_[0.01, 0.09]',
                           'case1_lambda_[0.1, 0.9]',
                           'case1_lambda_[1, 4]',
                           'case1_lambda_[0.01, 5]',
                           'case2_lambda_1',
                           'case2_lambda_[0.01, 0.09]',
                           'case2_lambda_[0.1, 0.9]',
                           'case2_lambda_[1, 4]',
                           'case2_lambda_[0.01, 5]'
                           ]

    data = datasets[int(sys.argv[1])]
    experiment_number = sys.argv[2]

    if experiment_number == 'ex1' or experiment_number == 'ex2':
        path_store = f'rolling_window_experiments_transformed/results/{data}/{experiment_number}/error_percentages'
    else:
        path_store = f'rolling_window_experiments/results/{data}/error_percentages'

    if not os.path.exists(path_store):
        os.makedirs(path_store)

    for model in models:
        # # one step ahead horizon
        for error_name in ['MSE']:
            errors_per_fc_type = {}
            ml_errors_per_fc_type = {}
            ml_method_errors = []
            ml_mean = {}
            run_errors(data, model, errors_per_fc_type, error_name)
            lowest_error_ml = np.min(ml_method_errors)

            for key, val in ml_mean.items():
                if val == lowest_error_ml:
                    errors_per_fc_type[key] = ml_errors_per_fc_type[key]

            sample_wise_error_df = pd.DataFrame(errors_per_fc_type)
            sample_wise_error_df.to_csv(
                f'rolling_window_experiments_transformed/results/{data}/{experiment_number}/error_percentages/{model}_sample_wise_errors.csv')
