import pandas as pd
import numpy as np

data = 'wikipedia'
base_model = 'arima'
validation_loss_scenario = 'case1'
ML_method_cases = ['lambda_1',
                   'lambda_[0.01, 0.09]',
                   'lambda_[0.1, 0.9]',
                   'lambda_[1, 4]',
                   'lambda_[0.01, 5]', ]

all_run_times = []
# read all running times
for method in ML_method_cases:
    filename = f'rolling_window_experiments/hyper_params/{data}/EXE_{base_model}_{validation_loss_scenario}_{method}.csv'
    dataframe_time = pd.read_csv(filename, index_col=0)

    all_run_times.append(dataframe_time)

all_run_times = pd.concat(all_run_times, axis=1)
mean_exe_time = pd.DataFrame(all_run_times.mean(axis=1))


train_time = []
exe_time = []
for idx, val in mean_exe_time.iterrows():
    # training time
    if int(idx) % 10 == 0:
        train_time.append(val)
    # execution time once trained
    else:
        exe_time.append(val)


print("Mean training time:", np.mean(train_time)/ 60 )
print("Mean execution time:", np.mean(exe_time)/ 60)

