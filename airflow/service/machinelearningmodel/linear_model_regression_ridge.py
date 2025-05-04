from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.logs import LogEntry

class ModelLinearRegressionRidge:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def ridge_regression_model(self, optimize_alpha=True):

        if optimize_alpha:
            alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            ridge_model = Ridge()
            
            param_grid = {'alpha': alphas}
            
            grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='r2')
            grid_search.fit(self.X_train, self.y_train.ravel())
            
            self.best_alpha = grid_search.best_params_['alpha']
            self.ridge_model = Ridge(alpha=self.best_alpha)
        else:
            self.ridge_model = Ridge(alpha=1.0)
        
        self.ridge_model.fit(self.X_train, self.y_train.ravel())
        self.y_pred_ridge = self.ridge_model.predict(self.X_test)
                
        return self.ridge_model
    
    def metris_model(self):
        
        mse_lr = metrics.mean_squared_error(self.y_test, self.y_pred_ridge)
        r2_lr = metrics.r2_score(self.y_test, self.y_pred_ridge)
        sqrt_mse_lr = np.sqrt(mse_lr)
        log = LogEntry("riged", mse_lr, r2_lr, sqrt_mse_lr, {'alfa' : self.best_alpha})
        return log.to_dict()
