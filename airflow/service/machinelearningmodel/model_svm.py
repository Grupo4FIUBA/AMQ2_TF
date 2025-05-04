import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn import metrics
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.logs import LogEntry

class ModelSVM:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.ravel()
        self.y_test = y_test.ravel()
        self.model = None
        self.params = {}

    def construction_regression_model_svm(self, optimize=True):
        if optimize:
            param_grid = {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }

            svr = SVR()
            grid_search = GridSearchCV(svr, param_grid, cv=3, n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            self.params = grid_search.best_params_
            self.model = SVR(**grid_search.best_params_)
        else:
            self.model = SVR(C=1, epsilon=0.1, kernel='rbf', gamma='scale')

        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

        return self.model


    def metris_model(self):
        mse = metrics.mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(self.y_test, self.predictions)
        r2 = metrics.r2_score(self.y_test, self.predictions)

        log = LogEntry("svm", mse, r2, rmse, {'params': self.params, 'mae': mae})
        
        return log.to_dict()
