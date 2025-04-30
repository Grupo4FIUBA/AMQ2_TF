import numpy as np
import os
import joblib
import logging
import boto3
import io
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from airflow.decorators import dag, task

# Definición de paths locales para modelos, logs y gráficos
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")
PLOT_PATH = os.path.join("plots")

# Aseguro que existan los directorios necesarios
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

# Nombre del bucket de MinIO donde están los datos y modelos
MINIO_BUCKET = 'data'

# Creo un cliente S3 apuntando a MinIO con las credenciales correspondientes
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url='http://S3:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123',
        region_name='us-east-1'
    )

# Configuro el logger para la etapa de validación
def setup_logger():
    logger = logging.getLogger("validation_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'validation.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

# Descargo un archivo .npy desde MinIO y lo cargo como un array de NumPy
def download_npy_from_minio(s3, key_name):
    with io.BytesIO() as f:
        s3.download_fileobj(MINIO_BUCKET, key_name, f)
        f.seek(0)
        array = np.load(f)
    return array

# Descargo un modelo previamente guardado en formato .pkl desde MinIO
def download_model_from_minio(s3, model_name):
    file_path = os.path.join(MODEL_PATH, model_name)
    s3.download_file(MINIO_BUCKET, model_name, file_path)
    with open(file_path, 'rb') as f:
        model = joblib.load(f)
    return model

# Genero y guardo los gráficos de dispersión y de residuales
def plot_and_save_predictions(y_test, y_pred, model_name):
    # Gráfico de dispersión real vs predicho
    scatter_path = os.path.join(PLOT_PATH, f"{model_name}_scatter.png")
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: y_test vs y_pred")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    # Histograma de residuales
    residual_path = os.path.join(PLOT_PATH, f"{model_name}_residuals.png")
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"{model_name}: Residual Histogram")
    plt.tight_layout()
    plt.savefig(residual_path)
    plt.close()

    return scatter_path, residual_path

# Defino el DAG de Airflow para validación
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

        # Lista de nombres de los modelos a validar
        models = ['Ridge', 'RandomForest', 'XGBoost', 'LightGBM']

        print("Comienza la validación...")

        # Descargo datos de test desde MinIO
        try:
            X_test = download_npy_from_minio(s3, 'X_test.npy')
            y_test = download_npy_from_minio(s3, 'y_test.npy')
            print("Datos de test descargados correctamente.")
        except Exception as e:
            logger.error(f"Error descargando datos de test: {e}")
            print(f"Error descargando datos de test: {e}")
            return

        # Configuro la URI de MLflow y obtengo el experimento existente
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment_name = "Modelos Regresion Entrenados"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        mlflow.autolog()

        if experiment is None:
            logger.error(f"Experimento {experiment_name} no encontrado.")
            return

        experiment_id = experiment.experiment_id

        # Itero sobre los modelos para cargarlos, predecir y registrar métricas
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

                # Registro de validación en MLflow
                with mlflow.start_run(run_name=f"{name}_validation", experiment_id=experiment_id):
                    mlflow.log_param("model_type", name)
                    mlflow.log_metric("val_r2", r2)
                    mlflow.log_metric("val_rmse", rmse)

                    # Genero y subo los gráficos al tracking server de MLflow
                    scatter_path, residual_path = plot_and_save_predictions(y_test, y_pred, name)
                    mlflow.log_artifact(scatter_path, artifact_path="plots")
                    mlflow.log_artifact(residual_path, artifact_path="plots")

            except Exception as e:
                logger.error(f"Error evaluando modelo {name}: {e}")
                print(f"Error evaluando modelo {name}: {e}")

    # Llamo a la función principal de validación
    validate_models()

# Instanciación del DAG de validación
validation_main_dag()
