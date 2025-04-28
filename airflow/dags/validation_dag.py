import numpy as np
import os
import joblib
import logging
import boto3
import io
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from airflow.decorators import dag, task

# Paths
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")
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
    logger = logging.getLogger("validation_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'validation.log'))
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

def download_model_from_minio(s3, model_name):
    file_path = os.path.join(MODEL_PATH, model_name)
    s3.download_file(MINIO_BUCKET, model_name, file_path)
    with open(file_path, 'rb') as f:
        model = joblib.load(f)
    return model

@dag(
    dag_id='validation_task_pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["validation", "minio", "autos"]
)
def validation_main_dag():

    @task
    def validate_models():
        s3 = get_s3_client()
        logger = setup_logger()

        models = ['Ridge', 'RandomForest', 'XGBoost', 'LightGBM']

        print("Comienza la validación...")

        try:
            X_test = download_npy_from_minio(s3, 'X_test.npy')
            y_test = download_npy_from_minio(s3, 'y_test.npy')
            print("Datos de test descargados correctamente.")
        except Exception as e:
            logger.error(f"Error descargando datos de test: {e}")
            print(f"Error descargando datos de test: {e}")
            return

        for name in models:
            file_name = name + ".pkl"

            try:
                model = download_model_from_minio(s3, file_name)
                print(f"Modelo {name} descargado y cargado correctamente.")
            except Exception as e:
                logger.error(f"Error al cargar modelo {name}: {e}")
                print(f"Error al cargar modelo {name}: {e}")
                continue

            print(f"Evaluando modelo {name}...")
            try:
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                logger.info(f"Modelo: {name} | R2: {r2:.4f} | RMSE: {rmse:.2f}")
                print(f"Modelo: {name} | R2: {r2:.4f} | RMSE: {rmse:.2f}")
            except Exception as e:
                logger.error(f"Error evaluando modelo {name}: {e}")
                print(f"Error evaluando modelo {name}: {e}")

    validate_models()

# Instanciar el DAG
validation_main_dag()
