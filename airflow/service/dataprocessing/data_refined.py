import pandas as pd
import time as tm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.const import DICC_FUEL, DICC_OWNER, DICC_PAY, DICC_TRANSMISSION
from config.s3 import AWSConnector

class DataRefined:

    def __init__(self, file_name:str, time):
        self.file_name=file_name
        self.time = time
        self.s3 = AWSConnector()
        self.dataframe=self.s3.read_parquet_from_s3("trusted", file_name)

    def convert_dummy(self):
        _columns=["fuel", "seller_type", "transmission", "owner"]
        self.dataframe_dummy=self.dataframe
        for item in _columns:
            self.dataframe_dummy=pd.get_dummies(self.dataframe_dummy, columns=[item], drop_first=True, dtype=int)
            
    def save_data_refined(self):
        file_name_parquet  = f"data_refined_dummy{self.time}.parquet"
        self.dataframe_dummy.to_parquet(f"/opt/airflow/data/refined/{file_name_parquet}")
        self.s3.send_to_s3("refined", file_name_parquet)
        return file_name_parquet

    def run(self):
        self.convert_dummy()
        return self.save_data_refined()


##datarefined=DataRefined("../../data/trusted/data_trusted_1746233836.8389091.parquet")
