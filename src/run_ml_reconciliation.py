import pandas as pd
from src.ml_reconcilation import MLReconcile

if __name__ == '__main__':
    data = 'prison'
    number_of_levels = 5
    model = 'ets'
    seed_value = 1234
    seed_runs = [1234, 3456, 2311, 8311, 5677]
    file_name = f'{data}_{model}'

    df_actual = pd.read_csv(f"input_data/{data}_actual.csv")
    df_fitted = pd.read_csv(f"forecasts/{file_name}_fitted.csv")
    df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv")
    hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                    'lambda': [0.1, 0.9], 'learning_rate': [0.0001, 0.1]}

    ml_model = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, hyper_params, seed_runs)
    forecasts_adjusted, best_hyperparam_config = ml_model.run_ml_reconciliation()
    best_hyperparam_config.to_csv(f'results/{file_name}_best_params.csv', index=False)

    for run in forecasts_adjusted:
        forecasts = forecasts_adjusted[run]['forecasts']
        model_history = forecasts_adjusted[run]['model_history']
        forecasts.to_csv(f'results/{file_name}_adjusted_forecasts_{run}.csv')
        model_history.to_csv(f'results/{file_name}_model_history_{run}.csv', index=False)
