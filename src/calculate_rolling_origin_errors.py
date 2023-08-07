import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
import numpy as np


def calculate_error(err_func, y_true, y_pred, error_list, ts_list, ts_idx, level_list):
    err = err_func(y_true, y_pred)
    error_list.append([ts_list[ts_idx], err, level_list[ts_idx]])


def calculate_errors_per_fc(data, fc_type, actual_test, model, iteration, error_name):
    errors = []
    if fc_type == 'base':
        df_forecasts = pd.read_csv(f"results/{data}/{model}_forecasts_{iteration}.csv", index_col=1).iloc[:, 1:]
    else:
        df_forecasts = pd.read_csv(f"results/hts/{data}/{model}_{fc_type}_{iteration}.csv", index_col=0)

    # iterate through each time series in hierarchy
    ts_names = actual_test.index.values
    levels = actual_test['Level'].values
    for ts in range(len(actual_test)):
        actual_test_ts = actual_test.iloc[ts: ts + 1, 1:].values[0]
        # forecasts
        ts_fc = df_forecasts.iloc[ts: ts + 1, :].values[0]
        if error_name == 'MSE':
            calculate_error(mean_squared_error, actual_test_ts, ts_fc, errors, ts_names, ts, levels)
        else:
            calculate_error(mean_absolute_error, actual_test_ts, ts_fc, errors, ts_names, ts, levels)

    all_errors = pd.DataFrame(errors)
    all_errors.columns = ['ts_name', 'error', 'level']

    ts_wise_errors = all_errors.iloc[:, 0:2]
    ts_wise_errors = ts_wise_errors.set_index('ts_name')
    
    all_errors = all_errors.set_index('ts_name')

    # get level wise error
    errors = all_errors.groupby(by=['level']).mean().reset_index().round(2)

    # level wise mean
    errors.index = LEVELS[data]
    overall_error = pd.DataFrame(all_errors[['error']].mean()).transpose()
    overall_error.index = ['Overall']
    errors = pd.concat([errors, overall_error]).round(2)
    return {'errors': errors, 'ts_wise_errors': ts_wise_errors}


def run_errors(data, model, errors_per_fc_type, errors_per_fc_type_median, error_name):
    samples = ROLLING_WINDOWS[data]
    for fc_type in FC_TYPE:
        if fc_type == 'mintsample' and (data == 'prison' or data == 'wikipedia'):
            continue
        else:
            sample_errors = []
            for sample in range(1, samples + 1):
                actual_test = pd.read_csv(f"results/{data}/test_{sample}.csv", index_col=1)
                error_dic = calculate_errors_per_fc(data, fc_type, actual_test, model, sample, error_name)
                sample_error = error_dic['errors']
                sample_errors.append(sample_error)

            all_samples = pd.concat(sample_errors)

            sample_wise_error = pd.concat(sample_errors, axis=1).drop(columns='level')
            sample_wise_error.columns = [f'sample {i}' for i in range(1, samples + 1)]
            sample_wise_error = sample_wise_error.transpose()['Overall'].values.tolist()
            if fc_type != 'base' and 'case' not in fc_type:
                errors_per_fc_type_dic[fc_type] = sample_wise_error
            if 'case' in fc_type:
                ml_errors_per_fc_type[fc_type] = sample_wise_error
                ml_method_errors.append(np.mean(sample_wise_error))  # ml method overall mean error
                ml_mean[fc_type] = np.mean(sample_wise_error)

            # this will give the mean across all samples for a given method
            mean_error = all_samples.groupby(all_samples.index).mean().reindex(sample_errors[0].index.values)
            median_error = all_samples.groupby(all_samples.index).median().reindex(sample_errors[0].index.values)
            filename = f'{model}_{fc_type}_{error_name}'

            mean_error.to_csv(f"results/errors/{data}/{filename}.csv")
            median_error.to_csv(f"results/errors/{data}/{filename}_median.csv")
            errors_per_fc_type.append(mean_error)
            errors_per_fc_type_median.append(median_error)


if __name__ == '__main__':
    ROLLING_WINDOWS = {'prison': 24,
                       'tourism': 120,
                       'labour': 60,
                       'wikipedia': 70}

    datasets = ['prison', 'labour', 'tourism', 'wikipedia']
    models = ['arima', 'ets', 'deepAR', 'waveNet']
    LEVELS = {'tourism': ['Australia', 'States', 'Regions'],
              'prison': ['Australia', 'State', 'Gender', 'Legal', 'Indigenous'],
              'labour': ['Total Employees', 'Main Occupation', 'Employment Status', 'Gender'],
              'wikipedia': ['Total', 'Access', 'Agent', 'Language', 'Purpose', 'Article']}

    FC_TYPE = ['base', 'bottomup', 'ols', 'wls', 'mintsample', 'mintshrink', 'erm',
               'case2_lambda_1',
               'case2_lambda_[0.01, 0.09]',
               'case2_lambda_[0.1, 0.9]',
               'case2_lambda_[1, 4]',
               'case2_lambda_[0.01, 5]',
               'case1_lambda_1',
               'case1_lambda_[0.01, 0.09]',
               'case1_lambda_[0.1, 0.9]',
               'case1_lambda_[1, 4]',
               'case1_lambda_[0.01, 5]',
               ]

    FC_TYPE_prison_wiki = ['base', 'bottomup', 'ols', 'wls', 'mintshrink', 'erm',
                           'case2_lambda_1',
                           'case2_lambda_[0.01, 0.09]',
                           'case2_lambda_[0.1, 0.9]',
                           'case2_lambda_[1, 4]',
                           'case2_lambda_[0.01, 5]',
                           'case1_lambda_1',
                           'case1_lambda_[0.01, 0.09]',
                           'case1_lambda_[0.1, 0.9]',
                           'case1_lambda_[1, 4]',
                           'case1_lambda_[0.01, 5]',
                           ]

    data = datasets[int(sys.argv[1])]
    model = models[int(sys.argv[2])]
    ml_method = sys.argv[3]
    error_name = 'MSE'

    if ml_method:
        if ml_method == 'case2':
            FC_TYPE = FC_TYPE[0:12]
            FC_TYPE_prison_wiki = FC_TYPE_prison_wiki[0:11]
        else:
            FC_TYPE = FC_TYPE[0:7]
            FC_TYPE.append(ml_method)

            FC_TYPE_prison_wiki = FC_TYPE_prison_wiki[0:6]
            FC_TYPE_prison_wiki.append(ml_method)
        
    print(FC_TYPE)

    path_store = f'results/errors/{data}/error_percentages'

    if not os.path.exists(path_store):
        os.makedirs(path_store)

    # # one step ahead horizon
    errors_per_fc_type = []  # 0 index corresponds to base errors -- for prison and wiki mintsample is not there
    errors_per_fc_type_median = []
    percentages = []
    percentages_median = []

    errors_per_fc_type_dic = {}
    ml_errors_per_fc_type = {}
    ml_method_errors = []
    ml_mean = {}

    run_errors(data, model, errors_per_fc_type, errors_per_fc_type_median, error_name)
    lowest_error_ml = np.min(ml_method_errors)

    # store sample wise errors for all methods
    for key, val in ml_mean.items():
        if val == lowest_error_ml:
            errors_per_fc_type_dic[key] = ml_errors_per_fc_type[key]

    sample_wise_error_df = pd.DataFrame(errors_per_fc_type_dic)
    sample_wise_error_df.to_csv(f'results/errors/{data}/error_percentages/{model}_sample_wise_errors.csv')

    # get percentage improvement over base forecasts
    for idx_method in range(1, len(errors_per_fc_type)):
        # mean
        percentage_improvement = ((errors_per_fc_type[0] - errors_per_fc_type[idx_method]) / errors_per_fc_type[
            0]) * 100
        percentage_improvement = percentage_improvement.iloc[:, 1:]

        # median
        percentage_improvement_median = ((errors_per_fc_type_median[0] - errors_per_fc_type_median[
            idx_method]) / errors_per_fc_type_median[
                                             0]) * 100
        percentage_improvement_median = percentage_improvement_median.iloc[:, 1:]
        if data == 'prison' or data == 'wikipedia':
            percentage_improvement.columns = [FC_TYPE_prison_wiki[idx_method]]
            percentage_improvement_median.columns = [FC_TYPE_prison_wiki[idx_method]]
        else:
            percentage_improvement.columns = [FC_TYPE[idx_method]]
            percentage_improvement_median.columns = [FC_TYPE[idx_method]]
        percentages.append(percentage_improvement)
        percentages_median.append(percentage_improvement_median)
    percentages = pd.concat(percentages, axis=1)
    percentages_median = pd.concat(percentages_median, axis=1)
    file_name = f'{model}_{error_name}'

    percentages.sort_values(by='Overall', axis=1, ascending=False).round(2).to_csv(
        f'{path_store}/{file_name}.csv')
    percentages_median.sort_values(by='Overall', axis=1, ascending=False).round(2).to_csv(
        f'{path_store}/{file_name}_median.csv')
