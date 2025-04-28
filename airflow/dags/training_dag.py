import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import logging
from datetime import datetime
from airflow.decorators import dag, task
import boto3
import io

# Paths
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")

# Asegurar carpetas necesarias
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

MINIO_BUCKET = 'data'

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url='http://S3:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123',
        region_name='us-east-1'
    )

def setup_logger():
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'training.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def download_npy_from_minio(s3, key_name):
    with io.BytesIO() as f:
        s3.download_fileobj(MINIO_BUCKET, key_name, f)
        f.seek(0)
        array = np.load(f)
    return array

@dag(
    dag_id='training_task_pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "minio", "autos"]
)
def training_main_dag():

    @task
    def training_main():
        s3 = get_s3_client()
        logger = setup_logger()

        # Descargar datasets
        X_train = download_npy_from_minio(s3, 'X_train.npy')
        y_train = download_npy_from_minio(s3, 'y_train.npy')

        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42)
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                file_name = name + ".pkl"
                file_path = os.path.join(MODEL_PATH, file_name)
                joblib.dump(model, file_path)
                s3.upload_file(file_path, MINIO_BUCKET, file_name)
                logger.info(f"Modelo {name} creado y subido exitosamente.")
                print(f"Modelo {name} creado y subido exitosamente.")
            except Exception as e:
                logger.error(f"Error al procesar el modelo {name}: {e}")
                print(f"Error al procesar el modelo {name}: {e}")

    training_main()

# Instanciar el DAG
training_main_dag()
