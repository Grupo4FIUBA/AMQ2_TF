from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
import sys, os
import time as tm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.dataprocessing.data_clean import DataClean
from service.dataprocessing.data_refined import DataRefined
from service.machinelearningmodel.model_orchestrator import ModelOrchestrator

@dag(
    default_args={
        'owner': 'airflow',
        'retries': 1,
    },
    description='Un DAG para procesar y entrenar modelos',
    schedule_interval=None,
    start_date=datetime(2025, 5, 3),
    catchup=False,
)
def my_project_dag():

    @task
    def clean_task():
        path = "/opt/airflow/data/raw/Car_details_V3.csv"
        time = str(tm.time()).replace(".","_")
        data_clean = DataClean(path, time)
        file_name = data_clean.run()
        return {'file_name': file_name, 'time': time}

    @task
    def refine_task(data_from_clean):
        file_name = data_from_clean.get('file_name')
        time = data_from_clean.get('time')
        path = file_name
        data_refined = DataRefined(path, time)
        file_name = data_refined.run()
        return {'file_name':file_name}
    
    @task
    def creation_model_task(data_from_refined):
        file_name = data_from_refined.get('file_name')
        path = file_name
        data_refined = ModelOrchestrator(path)
        diccionary = data_refined.run()
        print("=== DIccionario ===")
        print(f"Path: {diccionary}")
        

    data_to_send = clean_task()
    data_refined = refine_task(data_to_send)
    creation_model_task(data_refined)

my_project_dag_instance = my_project_dag()