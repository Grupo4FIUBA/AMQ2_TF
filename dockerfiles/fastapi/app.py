from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import dill
import numpy as np
import os
import boto3
from fastapi.middleware.cors import CORSMiddleware
from lightgbm import LGBMRegressor  # Modelo LightGBM para regresión
from xgboost import XGBRegressor  # Modelo XGBoost para regresión

# Crear directorios si no existen
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

MODEL_PATH = os.path.join("models")

# Inicializar la aplicación
app = FastAPI()

# Middleware para CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración del cliente S3 para MinIO
s3 = boto3.client(
    's3',
    endpoint_url='http://S3:9000',
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-east-1'
)

# Cargar modelos desde MinIO
MINIO_BUCKET = 'data'
model_files = {
    "model": "XGBoost.pkl",
    "encoder": "encoder.pkl",
    "scaler": "scaler.pkl"
}

for key, filename in model_files.items():
    local_path = os.path.join(MODEL_PATH, filename)
    try:
        #local_path = os.path.join("models", filename)
        print(f"Descargando '{filename}' desde bucket '{MINIO_BUCKET}'...")
        s3.download_file(MINIO_BUCKET, filename, local_path)
        with open(local_path, 'rb') as f:
            globals()[key] = joblib.load(f)
        print(f"'{filename}' cargado exitosamente.")
    except Exception as e:
        print(f"Error al procesar '{filename}': {e}")
        raise e  # o continuar si querés que cargue lo que pueda


# Modelo de datos esperado por el endpoint
class CarFeatures(BaseModel):
    year: int
    km_driven: int
    engine_cc: float
    max_power_bhp: float
    mileage_kmpl: float
    seats: int
    torque_nm: float
    torque_rpm: int
    owner_rank: int
    fuel: str
    seller_type: str
    transmission: str
    brand: str

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        # Datos categóricos
        cat_data = [[
            features.fuel,
            features.seller_type,
            features.transmission,
            features.brand
        ]]
        cat_encoded = encoder.transform(cat_data)

        # Datos numéricos
        numeric_data = [[
            features.year,
            features.km_driven,
            features.engine_cc,
            features.max_power_bhp,
            features.mileage_kmpl,
            features.seats,
            features.torque_nm,
            features.torque_rpm,
            features.owner_rank
        ]]
        numeric_scaled = scaler.transform(numeric_data)

        # Concatenar todos los datos
        full_input = np.hstack((numeric_scaled, cat_encoded))

        # Predicción
        prediction = model.predict(full_input)[0]
        return {"predicted_price": float(prediction)}
    
    except Exception as e:
        return {"error": str(e)}

# Ejecutar con: uvicorn main:app --host 0.0.0.0 --port 8000
