import numpy as np  # Para manejar arrays y matrices
import os  # Para operaciones con paths del sistema de archivos
import joblib  # Para guardar y cargar modelos de manera eficiente
import logging  # Para generar logs durante la ejecución
import boto3  # Cliente de AWS/MinIO
import io  # Para manejo de flujos de bytes en memoria
import mlflow  # Herramienta de tracking de experimentos de ML
import mlflow.sklearn  # Tracking para modelos sklearn
import mlflow.xgboost  # Tracking para modelos XGBoost
import mlflow.lightgbm  # Tracking para modelos LightGBM

from lightgbm import LGBMRegressor  # Modelo LightGBM para regresión
from xgboost import XGBRegressor  # Modelo XGBoost para regresión

from sklearn.linear_model import Ridge  # Modelo Ridge (regresión lineal con regularización)
from sklearn.ensemble import RandomForestRegressor  # Random Forest para regresión
from sklearn.metrics import mean_squared_error, r2_score  # Métricas de evaluación
from datetime import datetime  # Para definir fecha de inicio del DAG
from airflow.decorators import dag, task  # Decoradores para definir DAGs y tasks en Airflow

# Defino los paths donde se guardarán los modelos y logs localmente
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")

# Creo los directorios si no existen
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# Nombre del bucket en MinIO donde guardo/leo los archivos
MINIO_BUCKET = 'data'

# Devuelve un cliente de S3 configurado para MinIO
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url='http://S3:9000',  # URL interna del servicio MinIO
        aws_access_key_id='minio',  # Usuario
        aws_secret_access_key='minio123',  # Contraseña
        region_name='us-east-1'  # Región dummy requerida por boto3
    )

# Setea y devuelve un logger para dejar trazas del entrenamiento
def setup_logger():
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'training.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

# Descarga un archivo .npy desde MinIO, lo carga en memoria y lo devuelve como array
def download_npy_from_minio(s3, key_name):
    with io.BytesIO() as f:
        s3.download_fileobj(MINIO_BUCKET, key_name, f)
        f.seek(0)
        array = np.load(f)
    return array

# Definición del DAG principal de entrenamiento en Airflow
@dag(
    dag_id='training_task_pipeline',  # Nombre del DAG
    schedule_interval=None,  # Se ejecuta manualmente (sin programación periódica)
    start_date=datetime(2024, 1, 1),  # Fecha de inicio para validación del DAG
    catchup=False,  # No correr ejecuciones pasadas
    tags=["training", "minio", "autos"]  # Tags para categorizar el DAG
)
def training_main_dag():

    # Task de entrenamiento
    @task
    def training_main():
        s3 = get_s3_client()  # Inicializo cliente de MinIO
        logger = setup_logger()  # Inicializo el logger

        # Descargo los datos de entrenamiento desde MinIO
        X_train = download_npy_from_minio(s3, 'X_train.npy')
        y_train = download_npy_from_minio(s3, 'y_train.npy')

        # Diccionario con los modelos a entrenar
        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
            'LightGBM': LGBMRegressor(random_state=42)
        }

        # Seteo la URL del servidor de MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")

        # Obtengo o creo el experimento en MLflow
        experiment_name = "Modelos Regresion Entrenados"
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None or experiment.lifecycle_stage == 'deleted':
            # Si el experimento no existe o está eliminado, lo creamos de nuevo
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Creado nuevo experimento: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Usando experimento existente: {experiment_name}")

        # Activo autologging de MLflow (trackea automáticamente parámetros, métricas, etc.)
        mlflow.autolog()
        
        # Itero sobre todos los modelos definidos
        for name, model in models.items():
            try:
                # Inicio una nueva corrida en MLflow
                with mlflow.start_run(run_name=name, experiment_id=experiment_id):
                    model.fit(X_train, y_train)  # Entreno el modelo
                    y_pred = model.predict(X_train)  # Predigo sobre los mismos datos (train)

                    # Calculo métricas
                    r2 = r2_score(y_train, y_pred)
                    rmse = mean_squared_error(y_train, y_pred, squared=False)

                    # Loggeo parámetros y métricas manuales (por redundancia con autolog)
                    mlflow.log_param("model_type", name)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("rmse", rmse)

                    # Guardo el modelo entrenado localmente en formato .pkl
                    file_name = name + ".pkl"
                    file_path = os.path.join(MODEL_PATH, file_name)
                    joblib.dump(model, file_path)

                    # Subo el modelo entrenado a MinIO
                    s3.upload_file(file_path, MINIO_BUCKET, file_name)

                    # Registro el modelo en MLflow según el tipo correspondiente
                    if name == "XGBoost":
                        mlflow.xgboost.log_model(model, "model")
                    elif name == "LightGBM":
                        mlflow.lightgbm.log_model(model, "model")
                    else:
                        mlflow.sklearn.log_model(model, "model")

                    # Log de éxito
                    logger.info(f"Modelo {name} creado, subido y registrado en MLflow.")
                    print(f"Modelo {name} registrado en MLflow.")

            except Exception as e:
                # Log de error si falla el entrenamiento o registro del modelo
                logger.error(f"Error al procesar el modelo {name}: {e}")
                print(f"Error al procesar el modelo {name}: {e}")

    training_main()  # Llamada a la función dentro del DAG

# Instanciación del DAG para que Airflow lo pueda ejecutar
training_main_dag()
