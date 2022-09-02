import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm

from hyperopt import fmin, tpe, hp, Trials
from functools import partial
from hyperopt.pyll.base import scope
from construct_heirarchy import ConstructHierarchy


class MLReconcile:
    def __init__(self, seed_value, actual_data, fitted_data, forecasts, number_of_levels, seed_runs, hyper_params_tune,
                 best_hyper_params=None, tune_hyper_params=True, random_state=42, split_size=0.2,
                 validate_hf_loss=False, l1_regularizer=False, return_seed_forecast=False, tune_lambda=True,
                 remove_skip=False, saved_model = False, saved_models=[]):
        """
        Class initialization
        :param seed_value: seed value for tensorflow to achieve reproducibility
        :type seed_value: integer
        :param actual_data: pre processed actual data
        :type actual_data: pandas dataframe
        :param fitted_data: fitted values from the base model of interest
        :type fitted_data: pandas dataframe
        :param forecasts: forecasts from the base model
        :type forecasts: pandas dataframe
        :param number_of_levels: number of levels in the hierarchy
        :type number_of_levels: integer
        :param seed_runs: different seeds for the model to run
        :type: list of integers
        :param hyper_params_tune: hyper parameters to tune
        :type hyper_params_tune: dictionary with keys 'number_of_layers': integer, 'epochs': list, 'dropout_rate': list,
        'max_norm_value': list, 'reconciliation_loss_lambda': list, 'learning_rate': list
        :param best_hyper_params: best hyper parameters configuration of the model.
        :type best_hyper_params: dictionary with keys 'layers': integer, 'no_units_layer': list, 'epochs': int,
        'dropout_rate': float, 'max_norm_value': int, 'reconciliation_loss_lambda': float, 'learning_rate': float
        :param tune_hyper_params: whether to tune hyper parameters or not: default is True.
        If true hyper_params_tune should be provided. If False best_hyper_params should be provided
        :param random_state: seed value for the model
        :type random_state: integer
        :param split_size: data split size for validation
        :type split_size: float
        :param validate_hf_loss: True if the validation loss calculates the entire hierarchy error. Default is False
        :param l1_regularizer: True if the L1 regularizer needs to added to the loss function. Default is False
        """
        self.hierarchy = None
        self.actual_data = actual_data
        self.fitted_data = fitted_data
        self.forecasts = forecasts
        self.number_of_levels = number_of_levels
        self.actual_transpose = None
        self.fitted_transpose = None
        self.forecast_transpose = None
        self.random_state = random_state
        self.validation_split_size = split_size
        self.scaler = MinMaxScaler()
        self.model_history = None
        self.seed_runs = seed_runs
        self.hyper_params = None
        self.best_hyper_params = None
        self.validate_hf_loss = validate_hf_loss
        self.l1_regularizer = l1_regularizer
        self.return_seed_forecast = return_seed_forecast
        self.tune_lambda = tune_lambda
        self.remove_skip = remove_skip
        self.saved_models = saved_models
        self.run_saved_models = saved_model
        self._run_initialization(seed_value, tune_hyper_params, hyper_params_tune, best_hyper_params)

    def _transpose_data(self, dataframe):
        """
        Transpose dataframe - column names are time series names and rows are values
        :param dataframe: dataframe to transpose
        :type dataframe: pandas dataframe
        :return: transposed dataframe
        :rtype: pandas dataframe
        """
        dataframe_transpose = dataframe.iloc[:, 1:]
        dataframe_transpose.columns = [col_idx if col_idx != 0 else "" for col_idx in
                                       range(len(dataframe_transpose.columns))]
        dataframe_transpose = dataframe_transpose.set_index("").transpose()
        dataframe_transpose = dataframe_transpose.astype("float32")
        return dataframe_transpose

    def _run_initialization(self, seed_value, tune_hyper_params, hyper_params_tune, best_hyper_params):
        """
        Function to run when __init__ is called
        :param seed_value: seed value for tensorflow
        :type seed_value: integer
        :return: None
        :rtype: None
        """
        tf.random.set_seed(seed_value)  # set seed for tensorflow
        if tune_hyper_params:
            if hyper_params_tune:
                self.hyper_params = hyper_params_tune
            else:
                raise Exception("Please provide hyper_params_tune parameter if tune_hyper_params is True")
        else:
            if best_hyper_params:
                self.best_hyper_params = best_hyper_params
            else:
                if not self.run_saved_models:
                    raise Exception("Please provide best_hyper_params parameter if tune_hyper_params is False")

        self.actual_transpose = self._transpose_data(self.actual_data)
        self.fitted_transpose = self._transpose_data(self.fitted_data)
        self.forecast_transpose = self._transpose_data(self.forecasts)
        self.hierarchy = ConstructHierarchy(self.fitted_transpose,
                                            self.number_of_levels)  # construct time series hierarchy

    def _custom_loss(self, reconciliation_loss_lambda):
        """
        Custom loss function with bottom level forecast error + reconciliation loss
        :param reconciliation_loss_lambda: lambda value for reconciliation
        :type reconciliation_loss_lambda: float
        :return: calculated loss
        :rtype: float
        """
        bottom_level_ts, start_index_bottom_ts = self.hierarchy.get_bottom_level_ts_info()
        hierarchy = self.hierarchy.get_hierarchy_indexes()

        def loss(data, y_pred):
            rec_loss_list = list()
            all_predictions = y_pred[:, :bottom_level_ts]
            actual = data[:, start_index_bottom_ts:]
            # calculate the bottom time series loss
            loss_fn = tf.losses.MeanSquaredError()
            prediction_error = loss_fn(actual, all_predictions)

            # Reconciliation error across the hierarchy
            for key in hierarchy:
                higher_node_index = key  # higher node index
                lower_node_indexes = hierarchy[higher_node_index]  # bottom level nodes connected to higher node

                higher_node_ts = tf.reshape(data[:, higher_node_index], [-1, 1])  # get higher level node data

                lower_index = lower_node_indexes[0] - start_index_bottom_ts  # finding the index value starting from 0
                high_index = lower_node_indexes[-1] - start_index_bottom_ts

                lower_node_ts_pred = y_pred[:,
                                     lower_index: (high_index + 1)]  # get predictions for all bottom level nodes
                lower_node_ts_agg_pred = tf.reduce_sum(lower_node_ts_pred, axis=1,
                                                       keepdims=True)  # calculate the bottom up predictions
                recon_loss = tf.math.reduce_mean(tf.square(higher_node_ts - lower_node_ts_agg_pred))
                rec_loss_list.append(recon_loss)

            recon_loss_agg = tf.reduce_sum(rec_loss_list)  # sum all the losses
            final_loss = prediction_error + reconciliation_loss_lambda * recon_loss_agg
            if self.l1_regularizer:
                regularizer = tf.keras.regularizers.L1()
                return regularizer(final_loss)
            else:
                return final_loss

        return loss

    def build_and_compile_model(self, hyperparams):
        tf.keras.backend.clear_session()  # destroy previously built models
        bottom_level_ts, start_index_bottom_ts = self.hierarchy.get_bottom_level_ts_info()

        no_layers = hyperparams['no_layers']
        no_units_layer = hyperparams['no_units_layer']
        learning_rate = hyperparams['learning_rate']
        dropout_rate = hyperparams['dropout_rate']
        max_norm_value = hyperparams['max_norm_value']

        if self.tune_lambda:
            reconciliation_loss_lambda = hyperparams['reconciliation_loss_lambda']
        else:
            reconciliation_loss_lambda = 1

        inputs = keras.Input(shape=self.fitted_transpose.shape[1])
        last_output = inputs
        for layer in range(no_layers):
            layer_output = layers.Dense(no_units_layer[layer],
                                        kernel_constraint=max_norm(max_norm_value))(last_output)
            x = layers.BatchNormalization()(layer_output)
            x = keras.activations.relu(x)
            last_output = layers.Dropout(dropout_rate)(x)

        model_outputs = layers.Dense(bottom_level_ts)(last_output)

        if self.remove_skip:
            model = keras.Model(inputs=inputs, outputs=model_outputs)
        else:
            bottom_level_actual = (inputs[:, start_index_bottom_ts:] - tf.constant(
                self.scaler.min_[start_index_bottom_ts:])) / tf.constant(self.scaler.scale_[start_index_bottom_ts:])

            outputs = tf.add(model_outputs, bottom_level_actual)  # skip connection from input to bypass the model
            model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=self._custom_loss(reconciliation_loss_lambda=reconciliation_loss_lambda),
                      optimizer=tf.keras.optimizers.Adam(learning_rate))
        return model

    def split_data(self):
        X_train, X_valid, Y_train, Y_valid = train_test_split(self.fitted_transpose, self.actual_transpose,
                                                              test_size=self.validation_split_size,
                                                              random_state=self.random_state)
        self.scaler.fit(X_train)
        X_train = pd.DataFrame(self.scaler.transform(X_train))  # training set
        X_valid = pd.DataFrame(self.scaler.transform(X_valid))  # validation set
        X_train_test = pd.DataFrame(
            self.scaler.transform(self.fitted_transpose))  # training set during testing
        X_test = pd.DataFrame(self.scaler.transform(self.forecast_transpose))  # testing set
        return {'X_train': X_train, 'X_valid': X_valid, 'Y_train': Y_train, 'Y_valid': Y_valid,
                'X_train_test': X_train_test, 'X_test': X_test}

    def create_hyper_param_range(self, data_dic):
        layers_upper = self.hyper_params['number_of_layers']
        learning_rate = self.hyper_params['learning_rate']
        epochs = self.hyper_params['epochs']
        dropout_rate = self.hyper_params['dropout_rate']
        max_norm_value = self.hyper_params['max_norm_value']

        if self.tune_lambda:
            rec_lambda = self.hyper_params['reconciliation_loss_lambda']

        layers_list = []
        for i in range(1, (layers_upper + 1)):
            layer_dict = {
                'no_layers': i,
                'no_units_layer': [scope.int(hp.quniform('no_units_layer_' + str(i) + "_" + str(j), 1, 256, 1))
                                   for j in range(1, (i + 1))]
            }
            layers_list.append(layer_dict)

        param_space = {
            'learning_rate': hp.uniform('learning_rate', learning_rate[0], learning_rate[1]),
            'layers': hp.choice('layers', layers_list),
            'batch_size': scope.int(hp.quniform('batch_size', 1, data_dic['X_train'].shape[0], 1)),
            'epochs': scope.int(hp.quniform('epochs', epochs[0], epochs[1], 1)),
            'dropout_rate': hp.uniform('dropout_rate', dropout_rate[0], dropout_rate[1]),
            'max_norm_value': hp.uniform('max_norm_value', max_norm_value[0], max_norm_value[1]),
        }

        if self.tune_lambda:
            param_space['reconciliation_loss_lambda'] = hp.uniform('reconciliation_loss_lambda', rec_lambda[0],
                                                                   rec_lambda[1])
        return param_space

    def validate_model(self, hyperparams, data_dic):
        def val_loss_fn(y_actual, y_pred):
            bottom_level_index = self.hierarchy.get_bottom_level_index()
            if self.validate_hf_loss:  # if the validation loss is calculated to the whole hierarchy
                bottom_up_predictions = None
                hierarchy = self.hierarchy.get_hierarchy_indexes()
                hf_ts_indexes = sorted(list(hierarchy.keys()))
                bottom_level_ts, start_index_bottom_ts = self.hierarchy.get_bottom_level_ts_info()

                # Reconciliation error across the hierarchy
                for higher_node_index in hf_ts_indexes:
                    lower_node_indexes = hierarchy[higher_node_index]  # bottom level nodes connected to higher node
                    lower_index = lower_node_indexes[
                                      0] - start_index_bottom_ts  # finding the index value starting from 0
                    high_index = lower_node_indexes[-1] - start_index_bottom_ts

                    lower_node_ts_pred = y_pred[:,
                                         lower_index: (high_index + 1)]  # get predictions for all bottom level nodes
                    lower_node_ts_agg_pred = tf.reduce_sum(lower_node_ts_pred, axis=1,
                                                           keepdims=True)  # calculate the bottom up predictions
                    if higher_node_index == 0:
                        bottom_up_predictions = lower_node_ts_agg_pred
                    else:
                        # concat all bottom up forecasts for top time series
                        bottom_up_predictions = tf.concat([bottom_up_predictions, lower_node_ts_agg_pred], 1)
                # combine with the bottom level predictions
                bottom_up_predictions = tf.concat([bottom_up_predictions, y_pred], 1)
                validation_loss = np.mean(np.sqrt(tf.losses.MSE(y_actual.values, bottom_up_predictions)))
            else:
                validation_loss = np.mean(np.sqrt(tf.losses.MSE(y_actual.values[:, bottom_level_index:], y_pred)))
            return validation_loss

        hyperparams['no_layers'] = hyperparams['layers']['no_layers']
        hyperparams['no_units_layer'] = hyperparams['layers']['no_units_layer']

        hierarchical_model = self.build_and_compile_model(hyperparams)

        hierarchical_model.fit(
            data_dic['X_train'], data_dic['Y_train'],
            batch_size=hyperparams['batch_size'],
            epochs=hyperparams['epochs'],
            verbose=1
        )

        validation_predictions = hierarchical_model.predict(x=data_dic['X_valid'])
        val_loss = val_loss_fn(data_dic['Y_valid'], validation_predictions)

        return val_loss

    def train_model(self, hyperparams, data_dic):
        hyperparams['no_layers'] = int(hyperparams['layers'] + 1)
        hyperparams['no_units_layer'] = [
            int(hyperparams['no_units_layer_' + str(hyperparams['layers'] + 1) + "_" + str(j)])
            for j in range(1, (hyperparams['layers'] + 2))]
        hierarchical_model = self.build_and_compile_model(hyperparams)

        history = hierarchical_model.fit(
            data_dic['X_train_test'], self.actual_transpose,
            batch_size=int(hyperparams['batch_size']),
            epochs=int(hyperparams['epochs']),
            verbose=1
        )
        self.model_history = history
        return hierarchical_model

    def _get_bottom_up_forecasts(self, adjusted_forecasts):
        bottom_up_data = []
        start_index_bottom_ts = self.hierarchy.get_bottom_level_index()
        hierarchy = self.hierarchy.get_hierarchy_indexes()

        for key in hierarchy:
            top_level_node_index = key
            bottom_level_ts_indexes = hierarchy[top_level_node_index]
            lower_index = bottom_level_ts_indexes[0] - start_index_bottom_ts  # finding the index value starting from 0
            high_index = bottom_level_ts_indexes[-1] - start_index_bottom_ts
            top_level_fc_adjusted = adjusted_forecasts.iloc[lower_index:(high_index + 1), :].sum(axis=0)
            top_level_fc_adjusted = pd.DataFrame(top_level_fc_adjusted)
            top_level_fc_adjusted.columns = [self.hierarchy.get_ts_names()[key]]
            bottom_up_data.append(top_level_fc_adjusted.transpose())

        return pd.concat(bottom_up_data).append(adjusted_forecasts)

    def _train_model_with_seeds(self, data_dic):
        predictions = []
        model_history = []
        for run in range(len(self.seed_runs)):
            tf.random.set_seed(self.seed_runs[run])  # set seed for tensorflow for a run
            if not self.run_saved_models:
                ml_rec_model = self.train_model(self.best_hyper_params, data_dic)
                self.saved_models.append(ml_rec_model)
            else:
                # retrieve the model we saved in a previous rolling window iteration
                ml_rec_model = self.saved_models[run]

            # forward propagate the forecasts to get the adjusted forecasts
            adjusted_forecasts = pd.DataFrame(ml_rec_model.predict(x=data_dic['X_test']))
            adjusted_forecasts.columns = self.hierarchy.hierarchy_levels[self.number_of_levels]
            adjusted_forecasts = adjusted_forecasts.transpose()
            predictions.append(adjusted_forecasts)

            if not self.run_saved_models:
                model_history_per_run = pd.DataFrame(self.model_history.history)
                model_history_per_run.columns = [f'loss_run_{run + 1}']
                model_history.append(model_history_per_run)
        seed_forecasts = pd.concat(predictions)
        median_forecast = seed_forecasts.groupby(seed_forecasts.index).median()
        mean_forecast = seed_forecasts.groupby(seed_forecasts.index).mean()
        forecast_for_hierarchy_median = self._get_bottom_up_forecasts(median_forecast)
        forecast_for_hierarchy_mean = self._get_bottom_up_forecasts(mean_forecast)

        if not self.run_saved_models:
            return forecast_for_hierarchy_median, forecast_for_hierarchy_mean, pd.concat(model_history,
                                                                                         axis=1), seed_forecasts
        else:
            return forecast_for_hierarchy_median, forecast_for_hierarchy_mean, [], seed_forecasts

    def run_ml_reconciliation(self):
        # split the dataset
        print("====> splitting data")
        data_dic = self.split_data()

        # find optimal hyper parameters
        if self.hyper_params:
            random_state = np.random.default_rng(42)
            print("====> running hyper parameter optimization")
            trials = Trials()
            max_evals = 15
            if self.tune_lambda:
                lambda_val = self.hyper_params['reconciliation_loss_lambda']
                diff_ = lambda_val[1] - lambda_val[0]
                if diff_ > 1:
                    max_evals = 25
            self.best_hyper_params = fmin(
                fn=partial(self.validate_model, data_dic=data_dic),
                space=self.create_hyper_param_range(data_dic),
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals,
                rstate=random_state,
                show_progressbar = False
            )

            # run the model with best parameters
            print("====> best model parameters")
            print(self.best_hyper_params)

        print("====> Training ML reconciliation model")
        # train model with multiple seed values and retrieve the adjusted forecasts
        forecasts_median, forecast_mean, model_history, seed_forecast = self._train_model_with_seeds(data_dic)
        if self.return_seed_forecast:
            return forecasts_median, forecast_mean, model_history, pd.DataFrame.from_dict(self.best_hyper_params,
                                                                                          'index'), seed_forecast
        else:
            if self.run_saved_models:
                return forecasts_median, forecast_mean, None, None, self.saved_models
            else:
                return forecasts_median, forecast_mean, model_history, pd.DataFrame.from_dict(self.best_hyper_params,
                                                                                          'index'), self.saved_models