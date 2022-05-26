from gluonts.model.wavenet import WaveNetEstimator
from gluonts.mx.trainer import Trainer
from dnn_class import DNN


class WaveNet(DNN):
    def __init__(self, level, data_train, data_test, start_date, freq, horizon):
        super().__init__(level, data_train, data_test, start_date, freq, horizon)

    def train_model(self):
        estimator = WaveNetEstimator(freq=self.freq, prediction_length=self.horizon, trainer=Trainer(epochs=50))
        self.predictor = estimator.train(training_data=self.train_dataset)

    def run_model(self):
        self.train_model()
        forecasts_test = self.get_forecasts()
        self.create_samples_fitted()
        fitted_values = self.get_fitted_values()
        return fitted_values, forecasts_test
