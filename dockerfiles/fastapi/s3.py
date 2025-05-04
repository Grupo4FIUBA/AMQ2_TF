import boto3, os, sys, time
import pandas as pd
import io
import joblib
import pickle

class AWSConnector:

    def __init__(self):
        try:
            self.s3_client = boto3.client(
                's3',
                #endpoint_url='http://localhost:9000',
                aws_access_key_id='minio',
                aws_secret_access_key='minio123',
                region_name='us-east-1'
            )
            response = self.s3_client.list_objects_v2(Bucket="data")
            print(response)

        except Exception as e:
            print("Could not create AWS client")
            print(e)
            sys.exit()

    def read_pkl_from_s3(self, bucket, key):
        #places file in the Minio bucket

        #Now to load the pickled file
        response = self.s3_client.get_object(Bucket="data", Key=f"{bucket}/{key}")

        body = response['Body'].read()
        data = pickle.loads(body)

        #sample records
        print (data.head())

        #with io.BytesIO() as f:
        #    self.s3_client.download_fileobj("data", f"{bucket}/{key}", f)
        #    f.seek(0)
        #    modelo = joblib.load(f)
        #    return modelo