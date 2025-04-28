import re
import os
import io
import joblib
import logging
import boto3

import numpy as np
import pandas as pd

from datetime import datetime
from airflow.decorators import dag, task

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DATA_PATH = '/opt/airflow/dags/'
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# Configuración cliente S3 para MinIO
s3 = boto3.client(
    's3',
    endpoint_url='http://S3:9000',
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-east-1'
)

MINIO_BUCKET = 'data'

# Configuración de logging
logging.basicConfig(
    filename=os.path.join(LOG_PATH, 'preprocessing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Funciones auxiliares
def load_datasets(): # Funcion de carga de archivos plano
    datasets = {
        'car_data': pd.read_csv(os.path.join(DATA_PATH, 'car_data.csv')),
        'car_details': pd.read_csv(os.path.join(DATA_PATH, 'CAR_DETAILS_FROM_CAR_DEKHO.csv')),
        'car_details_v4': pd.read_csv(os.path.join(DATA_PATH, 'car_details_V4.csv')),
        'car_details_v3': pd.read_csv(os.path.join(DATA_PATH, 'Car_details_V3.csv')),
    }
    return datasets

def basic_cleaning(df):
    df_clean = df.copy()

    df_clean.drop_duplicates(inplace=True)

    df_clean['brand'] = df_clean['name'].str.split().str[0]
    df_clean['model'] = df_clean['name'].str.split(n=1).str[1]

    df_clean['engine_cc'] = df_clean['engine'].str.extract('(\d+)').astype(float)
    df_clean['max_power_bhp'] = df_clean['max_power'].str.extract('(\d+\.?\d*)').astype(float)
    df_clean['max_power_bhp'].replace(0, np.nan, inplace=True)

    def standardize_mileage(x):
        if pd.isna(x):
            return np.nan
        if 'km/kg' in str(x):
            return float(str(x).split()[0]) * 1.39
        return float(str(x).split()[0])

    df_clean['mileage_kmpl'] = df_clean['mileage'].apply(standardize_mileage)

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

    df_clean[['torque_nm', 'torque_rpm']] = pd.DataFrame(
        df_clean['torque'].apply(extract_torque_values).tolist(),
        index=df_clean.index
    )

    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }
    df_clean['owner_rank'] = df_clean['owner'].map(owner_map)

    return df_clean

def prepare_data(df_clean, target_column='selling_price'):
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = ['year', 'km_driven', 'engine_cc', 'max_power_bhp',
                    'mileage_kmpl', 'seats', 'torque_nm', 'torque_rpm', 'owner_rank']
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'brand']

    imputer = IterativeImputer(random_state=42)
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_train = encoder.fit_transform(X_train[categorical_cols])
    encoded_test = encoder.transform(X_test[categorical_cols])

    encoded_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=X_train.index)
    encoded_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols), index=X_test.index)

    X_train = pd.concat([X_train[numeric_cols], encoded_train], axis=1)
    X_test = pd.concat([X_test[numeric_cols], encoded_test], axis=1)

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

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

# DAG de Airflow
# ... (todos tus imports y configuraciones siguen igual)

# DAG de Airflow
@dag(
    dag_id='preprocessing_task_pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["preprocessing", "minio", "autos"]
)
def preprocessing_main_dag():

    @task()
    def load_data_task():
        datasets = load_datasets()
        return datasets['car_details_v3'].to_json()  # Convertir el DataFrame a JSON para pasarlo entre tasks

    @task()
    def basic_cleaning_task(df_json):
        df = pd.read_json(df_json)
        df_clean = basic_cleaning(df)
        return df_clean.to_json()

    @task()
    def prepare_data_task(df_clean_json):
        df_clean = pd.read_json(df_clean_json)
        X_train, X_test, y_train, y_test = prepare_data(df_clean)

        # Guardar en MinIO
        def upload_npy_to_minio(array, key_name):
            with io.BytesIO() as f:
                np.save(f, array)
                f.seek(0)
                s3.upload_fileobj(f, MINIO_BUCKET, key_name)

        upload_npy_to_minio(X_train, 'X_train.npy')
        upload_npy_to_minio(y_train, 'y_train.npy')
        upload_npy_to_minio(X_test, 'X_test.npy')
        upload_npy_to_minio(y_test, 'y_test.npy')

    # Definir el flujo secuencial
    df_json = load_data_task()
    df_clean_json = basic_cleaning_task(df_json)
    prepare_data_task(df_clean_json)

# Instanciar el DAG
# ... (todos tus imports y configuraciones siguen igual)

# DAG de Airflow
@dag(
    dag_id='preprocessing_task_pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["preprocessing", "minio", "autos"]
)
def preprocessing_main_dag():

    @task()
    def load_data_task():
        datasets = load_datasets()
        return datasets['car_details_v3'].to_json()  # Convertir el DataFrame a JSON para pasarlo entre tasks

    @task()
    def basic_cleaning_task(df_json):
        df = pd.read_json(df_json)
        df_clean = basic_cleaning(df)
        return df_clean.to_json()

    @task()
    def prepare_data_task(df_clean_json):
        df_clean = pd.read_json(df_clean_json)
        X_train, X_test, y_train, y_test = prepare_data(df_clean)

        # Guardar en MinIO
        def upload_npy_to_minio(array, key_name):
            with io.BytesIO() as f:
                np.save(f, array)
                f.seek(0)
                s3.upload_fileobj(f, MINIO_BUCKET, key_name)

        upload_npy_to_minio(X_train, 'X_train.npy')
        upload_npy_to_minio(y_train, 'y_train.npy')
        upload_npy_to_minio(X_test, 'X_test.npy')
        upload_npy_to_minio(y_test, 'y_test.npy')

    # Definir el flujo secuencial
    df_json = load_data_task()
    df_clean_json = basic_cleaning_task(df_json)
    prepare_data_task(df_clean_json)

# Instanciar el DAG
preprocessing_main_dag()


