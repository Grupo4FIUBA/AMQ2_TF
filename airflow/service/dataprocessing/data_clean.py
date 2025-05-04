import pandas as pd
import numpy as np
import re
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.s3 import AWSConnector


class DataClean:

    def __init__(self, path:str, time):
        self.path=path
        self.time = time
        self.dataframe=pd.read_csv(self.path)
        self.s3 = AWSConnector()


    def standar_columns(self):
        self.dataframe.columns=(self.dataframe.columns.str.strip().str.lower().str.replace(' ','_',regex=False).str.replace(r'[^\w]','',regex=True))
    
    def convert_column_float(self):
        columns=["selling_price", "km_driven"]
        self.dataframe[columns]=self.dataframe[columns].astype(float)
        print(self.dataframe.head(5))

    def transformation_columns(self):
        columns=["mileage", "engine", "max_power"]
        for item in columns:
            self.dataframe[item]=self.dataframe[item].str.extract(r'(\d+(?:\.\d+)?)').astype(float)

    def convert_torque(self):
        self.dataframe[['torque_nm', 'torque_rpm']] = pd.DataFrame(self.dataframe['torque'].apply(self.extract_torque_values).tolist(),index=self.dataframe.index)

    def extract_torque_values(self, x):
        if pd.isna(x):
            return np.nan, np.nan

        x_lower = str(x).lower().replace(',', '').strip()

        torque_patterns = [
            (r'(\d+\.?\d*)\s*nm', 1),
            (r'(\d+\.?\d*)\s*kgm', 9.80665),
            (r'(\d+\.?\d*)@', 9.80665),
            (r'(\d+\.?\d*)\s*at', 9.80665)
        ]

        rpm_patterns = [
            r'\(.*@\s*(\d+).*\)',
            r'@\s*(\d+)(?:\s*-\s*\d+)?\s*(?:rpm)?',
            r'rpm\s*@\s*(\d+)(?:\s*-\s*\d+)?',
            r'at\s*(\d+)(?:\s*-\s*\d+)?\s*(?:rpm)?',
            r'(\d+)(?:\s*-\s*\d+)?\s*rpm'
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
                rpm_match = re.search(r'(\d+)', match.group(0))
                if rpm_match:
                    rpm_value = int(rpm_match.group(1))
                break
            
        return torque_value, rpm_value
    
    def clean_nulls(self):
        self.dataframe=self.dataframe.dropna()
        self.dataframe=self.dataframe.drop(["torque", "name"], axis=1)
        print(self.dataframe.head(60))
        print(self.dataframe.info())

    def save_transformed_data(self):
        file_name = f"data_trusted_{self.time}.parquet"
        path = f"/opt/airflow/data/trusted/{file_name}"
        self.dataframe.to_parquet(path)
        self.s3.send_to_s3("trusted", file_name)
        return file_name

    def run(self):
        self.standar_columns()
        self.convert_column_float()
        self.transformation_columns()
        self.convert_torque()
        self.clean_nulls()
        return self.save_transformed_data()

