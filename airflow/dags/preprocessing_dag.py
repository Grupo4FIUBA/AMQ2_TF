# === Librerías base ===
import re  # Expresiones regulares para limpieza y extracción de texto
import os  # Operaciones sobre el sistema de archivos
import io  # Manejo de flujos de datos en memoria
import joblib  # Para guardar y cargar objetos de sklearn
import logging  # Logging centralizado del proceso
import boto3  # Cliente AWS S3 compatible con MinIO

# === Librerías científicas ===
import numpy as np  # Manipulación de arrays
import pandas as pd  # Manipulación de datos tabulares

from datetime import datetime  # Para definir fecha de inicio del DAG
from airflow.decorators import dag, task  # Decoradores para definir DAGs y tareas en Airflow

# === Sklearn para procesamiento y modelado ===
from sklearn.model_selection import train_test_split  # División entrenamiento/prueba
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Normalización y codificación
from sklearn.experimental import enable_iterative_imputer  # Habilito imputador experimental
from sklearn.impute import IterativeImputer  # Imputador más sofisticado que usa modelos iterativos
from sklearn.linear_model import Ridge  # Modelo de regresión lineal regularizada
from sklearn.ensemble import RandomForestRegressor  # Regresor basado en árboles
from sklearn.decomposition import PCA  # Reducción de dimensión
from sklearn.manifold import TSNE  # Visualización no lineal de alta dimensión
from sklearn.cluster import DBSCAN  # Algoritmo de clustering no supervisado

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas de regresión

# === Configuraciones de paths ===
DATA_PATH = '/opt/airflow/dags/'  # Carpeta donde se espera que estén los datos originales
MODEL_PATH = os.path.join("models")  # Carpeta para guardar modelos y artefactos
LOG_PATH = os.path.join("logs")  # Carpeta para logs

# Aseguro que existan las carpetas necesarias
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# === Cliente S3 apuntando a MinIO local ===
s3 = boto3.client(
    's3',
    endpoint_url='http://S3:9000',
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-east-1'
)

MINIO_BUCKET = 'data'  # Nombre del bucket en MinIO

# === Configuración del sistema de logs ===
logging.basicConfig(
    filename=os.path.join(LOG_PATH, 'preprocessing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Funciones auxiliares ===

# Cargo todos los datasets desde MinIO en memoria usando BytesIO
def load_datasets():
    datasets = {}
    files = {
        'car_data': 'car_data.csv',
        'car_details': 'CAR_DETAILS_FROM_CAR_DEKHO.csv',
        'car_details_v4': 'car_details_V4.csv',
        'car_details_v3': 'Car_details_V3.csv',
    }

    for key, filename in files.items():
        with io.BytesIO() as f:
            s3.download_fileobj(MINIO_BUCKET, filename, f)  # Descargo el archivo a memoria
            f.seek(0)
            datasets[key] = pd.read_csv(f)  # Leo directamente desde el buffer en memoria
    
    return datasets

# Función que realiza limpieza básica de columnas y extracción de variables
def basic_cleaning(df):
    df_clean = df.copy()

    df_clean.drop_duplicates(inplace=True)  # Elimino duplicados exactos

    # Extraigo la marca (brand) y el modelo del campo "name"
    df_clean['brand'] = df_clean['name'].str.split().str[0]
    df_clean['model'] = df_clean['name'].str.split(n=1).str[1]

    # Extraigo la cilindrada (cc) del campo "engine"
    df_clean['engine_cc'] = df_clean['engine'].str.extract('(\d+)').astype(float)

    # Extraigo la potencia máxima del campo "max_power"
    df_clean['max_power_bhp'] = df_clean['max_power'].str.extract('(\d+\.?\d*)').astype(float)
    df_clean['max_power_bhp'].replace(0, np.nan, inplace=True)  # Reemplazo ceros por NaN

    # Convierte el consumo de km/kg a km/l (1 kg de GNC ~ 1.39 litros de nafta)
    def standardize_mileage(x):
        if pd.isna(x):
            return np.nan
        if 'km/kg' in str(x):
            return float(str(x).split()[0]) * 1.39
        return float(str(x).split()[0])

    df_clean['mileage_kmpl'] = df_clean['mileage'].apply(standardize_mileage)

    # Extrae valor y RPM del torque, convirtiendo kgm a Nm
    def extract_torque_values(x):
        if pd.isna(x):
            return np.nan, np.nan
        x_lower = str(x).lower().strip()
        
        torque_patterns = [
            (r'(\d+\.?\d*)\s*Nm', 1),
            (r'(\d+\.?\d*)\s*kgm', 9.80665)
        ]
        
        rpm_patterns = [
            r'@\s*(\d+)',
            r'\((\d+)\s*rpm\)'
        ]
        
        torque_value = np.nan
        rpm_value = np.nan
        
        for pattern, factor in torque_patterns:
            match = re.search(pattern, x_lower)
            if match:
                torque_value = float(match.group(1)) * factor
                break
        
        for pattern in rpm_patterns:
            match = re.search(pattern, x_lower)
            if match:
                rpm_value = float(match.group(1))
                break
        
        return torque_value, rpm_value

    # Aplico la extracción a la columna torque
    df_clean[['torque_nm', 'torque_rpm']] = pd.DataFrame(
        df_clean['torque'].apply(extract_torque_values).tolist(),
        index=df_clean.index
    )

    # Map de propietarios a valores ordinales
    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }
    df_clean['owner_rank'] = df_clean['owner'].map(owner_map)

    return df_clean

# Preparo los datos: imputación, codificación, escalado
def prepare_data(df_clean, target_column='selling_price'):
    # Separo variables dependientes e independientes
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Divido entre entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Columnas numéricas y categóricas que vamos a procesar
    numeric_cols = ['year', 'km_driven', 'engine_cc', 'max_power_bhp',
                    'mileage_kmpl', 'seats', 'torque_nm', 'torque_rpm', 'owner_rank']
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'brand']

    # Imputación de valores faltantes en numéricas
    imputer = IterativeImputer(random_state=42)
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    # Codificación one-hot de variables categóricas
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_train = encoder.fit_transform(X_train[categorical_cols])
    encoded_test = encoder.transform(X_test[categorical_cols])

    encoded_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=X_train.index)
    encoded_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols), index=X_test.index)

    # Concateno numéricas y codificadas
    X_train = pd.concat([X_train[numeric_cols], encoded_train], axis=1)
    X_test = pd.concat([X_test[numeric_cols], encoded_test], axis=1)

    # Escalado estándar para normalizar las numéricas
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Guardo los objetos de preprocesamiento y los subo a MinIO
    try:
        encoder_path = os.path.join(MODEL_PATH, 'encoder.pkl')
        joblib.dump(encoder, encoder_path)
        s3.upload_file(encoder_path, MINIO_BUCKET, 'encoder.pkl')
        logging.info("Archivo encoder.pkl creado y subido.")
        
        scaler_path = os.path.join(MODEL_PATH, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        s3.upload_file(scaler_path, MINIO_BUCKET, 'scaler.pkl')
        logging.info("Archivo scaler.pkl creado y subido.")
    except Exception as e:
        logging.error(f"Error al guardar los objetos: {e}")

    return X_train, X_test, y_train, y_test

# === Definición del DAG en Airflow ===
@dag(
    dag_id='preprocessing_task_pipeline',
    schedule_interval=None,  # No se ejecuta automáticamente, se lanza manualmente
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["preprocessing", "minio", "autos"]
)
def preprocessing_main_dag():

    @task()
    def load_data_task():
        datasets = load_datasets()
        return datasets['car_details_v3'].to_json()  # Paso los datos como JSON entre tasks

    @task()
    def basic_cleaning_task(df_json):
        df = pd.read_json(df_json)
        df_clean = basic_cleaning(df)
        return df_clean.to_json()

    @task()
    def prepare_data_task(df_clean_json):
        df_clean = pd.read_json(df_clean_json)
        X_train, X_test, y_train, y_test = prepare_data(df_clean)

        # Subida a MinIO de arrays como archivos .npy
        def upload_npy_to_minio(array, key_name):
            with io.BytesIO() as f:
                np.save(f, array)
                f.seek(0)
                s3.upload_fileobj(f, MINIO_BUCKET, key_name)

        upload_npy_to_minio(X_train, 'X_train.npy')
        upload_npy_to_minio(y_train, 'y_train.npy')
        upload_npy_to_minio(X_test, 'X_test.npy')
        upload_npy_to_minio(y_test, 'y_test.npy')

    # Encadenamiento de tareas
    df_json = load_data_task()
    df_clean_json = basic_cleaning_task(df_json)
    prepare_data_task(df_clean_json)

# Instancio el DAG para que Airflow lo registre
preprocessing_main_dag()
