from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys, os
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machinelearningmodel.linear_model_regression_ridge import ModelLinearRegressionRidge
from machinelearningmodel.linear_model_regression import ModelLinearRegression
from machinelearningmodel.model_xgb_regressor import ModelXGBRegressor
from machinelearningmodel.model_svm import ModelSVM
from config.s3 import AWSConnector

class ModelOrchestrator: 

    def __init__(self, file_name:str):
        self.file_name=file_name
        self.s3 = AWSConnector()
        self.dataframe=self.s3.read_parquet_from_s3("refined", file_name)

    def save_model(self, file_name, model):
        path_model=f"/opt/airflow/data/model/{file_name}.pkl"
        joblib.dump(model, path_model)
        self.s3.send_to_s3("model", f"{file_name}.pkl")
        return path_model


    def standardization_data_linear(self):
        # Columnas no significativas -> 'fuel_Petrol', 'torque_nm', 'owner_Fourth & Above Owner', 'owner_Third Owner'
        self.dataframe["log_selling_price"] = np.log1p(self.dataframe["selling_price"])
        self.dataframe["log_km_driven"] = np.log1p(self.dataframe["km_driven"])
        X_cols = list(set(self.dataframe.columns)-set(['log_selling_price','km_driven','selling_price','torque_nm','mileage','seats', 'torque_rpm','engine','max_power']))
        y_col = ['log_selling_price']

        self.X_cols = X_cols
        
        self.X = self.dataframe[X_cols].values
        
        self.y = self.dataframe[y_col].values

        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(self.X,self.y)
        sc_x = StandardScaler().fit(self.X)
        sc_y = StandardScaler().fit(self.y)

        self.X_train = sc_x.transform(self.X_train_)
        self.X_test = sc_x.transform(self.X_test_)
        self.y_train = sc_y.transform(self.y_train_)
        self.y_test = sc_y.transform(self.y_test_)
        
        
        self.y_train_r = sc_y.transform(self.y_train_.reshape(-1, 1))
        self.y_test_r = sc_y.transform(self.y_test_.reshape(-1, 1))
        
        self.X_cols = X_cols
        
        self.X_d = self.dataframe[X_cols]
        self.y_d = self.dataframe[y_col]
        
    
       
    def exectuion_model(self):
        model_ridge = ModelLinearRegressionRidge( self.X_train, self.X_test, self.y_train, self.y_test)
        pk_model_ridge = model_ridge.ridge_regression_model()
        log_metric_ridge = model_ridge.metris_model()
        
        model_regression_linear = ModelLinearRegression( self.X_train, self.X_test, self.y_train, self.y_test)
        pk_model_linear = model_regression_linear.construction_regression_model()
        log_metric_linear = model_regression_linear.metris_model()
        
        model_xgb_regressor = ModelXGBRegressor( self.X_d, self.y_d)
        pk_model_xgb = model_xgb_regressor.construction_regression_model_xgboost()
        log_metric_xgb = model_xgb_regressor.cross_valcore_model()
        
        model_svm_regressor = ModelSVM( self.X_train, self.X_test, self.y_train_r, self.y_test_r)
        pk_model_svm = model_svm_regressor.construction_regression_model_svm()
        log_metric_svm = model_svm_regressor.metris_model()

        path_ridge=self.save_model("model_ridge", pk_model_ridge)
        path_regression_linear=self.save_model("model_regression_linear", pk_model_linear)
        path_xgb_regressor=self.save_model("model_xgb_regressor", pk_model_xgb)
        path_svm_regressor=self.save_model("model_svm_regressor", pk_model_svm)

        return {
            'ridge': {
                'path': path_ridge,
                'log_metric': log_metric_ridge
            },
            'linear': {
                'path': path_regression_linear,
                'log_metric': log_metric_linear
            },
            'xgb': {
                'path': path_xgb_regressor,
                'log_metric': log_metric_xgb
            },
            'svm': {
                'path': path_svm_regressor,
                'log_metric': log_metric_svm
            }
        }


    def run(self):
        self.standardization_data_linear()
        diccionary = self.exectuion_model()
        return diccionary