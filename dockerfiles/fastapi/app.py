from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import boto3
import joblib
import io
import sys, os
from typing import List 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3 import AWSConnector

class InputData(BaseModel):
    features: List[float]

class APIModelos:

    def __init__(self):
        self.app = FastAPI(title="API de Modelos de ML")
        self.bucket = 'mi-bucket-modelos'
        self.s3=AWSConnector()
        self.model_regression_linear=self.s3.read_pkl_from_s3("model", "model_regression_linear.pkl")
        self.model_ridge=self.s3.read_pkl_from_s3("model", "model_ridge.pkl")
        self.model_svm_regressor=self.s3.read_pkl_from_s3("model", "model_svm_regressor.pkl")
        self.model_xgb_regressor=self.s3.read_pkl_from_s3("model", "model_xgb_regressor.pkl")
        self.modelos = {
            'ridge': 'modelo_ridge.pkl',
            'linear': 'modelo_linear.pkl',
            'svr': 'modelo_svr.pkl',
            'xgb': 'modelo_xgb.pkl'
        }
        # Rutas
        self.definir_rutas()

    def definir_rutas(self):
        @self.app.post("/predecir/{modelo}")
        def predecir(modelo: str, data: InputData):
            if modelo not in self.modelos:
                raise HTTPException(status_code=404, detail="Modelo no disponible")

            try:
                modelo_sklearn = self.obtener_modelo(modelo)
                entrada = np.array(data.features).reshape(1, -1)
                prediccion = modelo_sklearn.predict(entrada)
                return {"modelo": modelo, "prediccion": float(prediccion[0])}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def obtener_modelo(self, nombre_modelo: str):
        
        nombre_archivo = self.modelos[nombre_modelo]
        try:
            with io.BytesIO() as f:
                self.s3.download_fileobj(self.bucket, nombre_archivo, f)
                f.seek(0)
                modelo = joblib.load(f)
                self.cache[nombre_modelo] = modelo
                return modelo
        except Exception as e:
            raise RuntimeError(f"Error al cargar modelo desde S3: {e}")

# Instancia la clase
api_modelos = APIModelos()
app = api_modelos.app