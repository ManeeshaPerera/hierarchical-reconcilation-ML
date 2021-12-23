import pandas as pd
from src.ml_reconcilation import MLReconcile

if __name__ == '__main__':
    data = 'prison'
    number_of_levels = 5
    model = 'arima'
    seed_value = 1234
    file_name = f'{data}_{model}'

    df_actual = pd.read_csv(f"input_data/{data}_actual.csv")
    df_fitted = pd.read_csv(f"forecasts/{file_name}_fitted.csv")
    df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv")
    hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                    'lambda': [0.1, 0.9], 'learning_rate': [0.0001, 0.1]}

    ml_model = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, hyper_params)
    forecasts_adjusted, best_hyperparam_config, model_history = ml_model.run_ml_reconciliation()
    forecasts_adjusted.to_csv(f'results/{file_name}_adjusted_forecasts.csv')
    best_hyperparam_config.to_csv(f'results/{file_name}_best_params.csv', index=False)
    model_history.to_csv(f'results/{file_name}_model_history.csv', index=False)
