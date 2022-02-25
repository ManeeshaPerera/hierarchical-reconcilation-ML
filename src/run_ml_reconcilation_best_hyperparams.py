import pandas as pd
from src.ml_reconcilation import MLReconcile

if __name__ == '__main__':
    levels_in_hierarchy = {'prison': 5, 'tourism': 3}
    data = 'prison'
    number_of_levels = levels_in_hierarchy[data]
    model = 'ets'
    seed_value = 1234
    # seed_runs = [1234]
    seed_runs = [1234, 3456, 2311, 8311, 5677]
    file_name = f'{data}_{model}'

    df_actual = pd.read_csv(f"input_data/{data}_actual.csv")
    df_fitted = pd.read_csv(f"forecasts/{file_name}_fitted.csv")
    df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv")

    # read best hyper paramters
    # hyper_params = pd.read_csv(f"results/{file_name}_best_params_case1.csv", index_col=0)
    # previous best params found
    hyper_params = pd.read_csv(f"results/validation_results_experiments/tourism_best_params_prev.csv",
                               index_col=0)
    best_hyper_params = {}
    no_units_layer = []

    for param, value in hyper_params.itertuples():
        if param == 'no_units_layer':
            continue
        if 'no_units_layer' in param:
            no_units_layer.append(int(float(value)))
        if param == 'no_layers':
            continue
        if param == 'layers':
            best_hyper_params[param] = int(float(value))
        else:
            best_hyper_params[param] = float(value)
    best_hyper_params['no_units_layer'] = no_units_layer

    # CASE 1 - validation loss for bottom level
    ml_model_case1 = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, seed_runs,
                                 hyper_params_tune=None,
                                 tune_hyper_params=False, best_hyper_params=best_hyper_params)
    forecasts_adjusted_case1, forecasts_adjusted_case1_mean, model_history_case1, best_hyper_params_case1 = ml_model_case1.run_ml_reconciliation()

    forecasts_adjusted_case1.to_csv(f'results/{file_name}_adjusted_forecasts_seeds_prev_params.csv')
    forecasts_adjusted_case1_mean.to_csv(f'results/{file_name}_adjusted_forecasts_seeds_prev_params_mean.csv') # mean is not needed if it's one seed
    # model_history_case1.to_csv(f'results/{file_name}_model_history_one_seed_prev_params.csv')
