import boto3, os, sys, time
import pandas as pd
import io
import joblib

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

    def read_pkl_from_s3(self, bucket, key):
        with io.BytesIO() as f:
            self.s3_client.download_fileobj("data", f"{bucket}/{key}", f)
            f.seek(0)
            modelo = joblib.load(f)
            return modelo