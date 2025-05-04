from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.logs import LogEntry
class ModelLinearRegression:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def construction_regression_model(self):
        model = LinearRegression()
        model.fit(self.X_train,self.y_train)
        self.y_pred = model.predict(self.X_test)
        return model
    
    def metris_model(self):
        mse_lr = metrics.mean_squared_error(self.y_test, self.y_pred)
        r2_lr = metrics.r2_score(self.y_test, self.y_pred)
        sqrt_mse_lr = np.sqrt(mse_lr)
        log = LogEntry("riged", mse_lr, r2_lr, sqrt_mse_lr, {})
        return log.to_dict()
