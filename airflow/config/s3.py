import boto3, os, sys, time
import pandas as pd
import io

class AWSConnector:

    def __init__(self):
        try:
            self.s3_client = boto3.client(
                's3',
                #endpoint_url='http://S3:9000',
                aws_access_key_id='minio',
                aws_secret_access_key='minio123',
                region_name='us-east-1'
            )
        except Exception as e:
            print("Could not create AWS client")
            print(e)
            sys.exit()

    def send_to_s3(self, folder, file_name):

        destination_folder = f"{folder}"
        file_path = f"/opt/airflow/data/{folder}/{file_name}"

        try:
            with open(file_path, "rb") as file_manager:
                self.s3_client.upload_fileobj(
                    file_manager,
                    "data",
                    f"{destination_folder}/{file_name}",
                )
                print(
                    f"File uploaded successfully to S3 at s3://data/{destination_folder}/{file_name}"
                )
        except Exception as e:
            print(f"An error occurred when moving the file {file_name}")
            raise e
        
        finally:
            os.remove(file_path)

    def read_parquet_from_s3(self, bucket, key):
        with io.BytesIO() as f:
            self.s3_client.download_fileobj("data", f"{bucket}/{key}", f)
            f.seek(0)
            return pd.read_parquet(f)
            