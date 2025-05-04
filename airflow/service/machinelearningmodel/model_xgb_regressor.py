import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBRegressor
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.logs import LogEntry

class ModelXGBRegressor:
    
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        self.model = None
        self.params = {}

    def construction_regression_model_xgboost(self, optimize=True):
        scaler = StandardScaler()  
        self.x_scaled = scaler.fit_transform(self.x) 

        if optimize:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'min_child_weight': [1, 3],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'alpha': [0, 0.1],
                'lambda': [1.0] 
            }

            xgb = XGBRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1, verbose=1)
            grid_search.fit(self.x_scaled, self.y) 
            self.params = grid_search.best_params_
            self.model = XGBRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
        else:
            self.model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        self.model.fit(self.x_scaled, self.y) 
        return self.model

    def cross_valcore_model(self, cv=5):
        scaler = StandardScaler()
        self.x_scaled = scaler.fit_transform(self.x)  

        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error'
        }

        scores = cross_validate(self.model, self.x_scaled, self.y, cv=cv, scoring=scoring, n_jobs=-1)

        r2_scores = scores['test_r2']
        mse_scores = -scores['test_neg_mse']
        rmse_scores = np.sqrt(mse_scores)

        log = LogEntry("xgboost_cv", 
                       float(np.mean(mse_scores)),
                       float(np.mean(r2_scores)),
                       float(np.mean(rmse_scores)),
                       self.params
        )
        return log.to_dict()
