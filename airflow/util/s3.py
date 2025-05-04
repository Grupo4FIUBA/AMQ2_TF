from config.s3 import AWSConnector
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class S3Uploader:

    def __init__(self) -> None:
        self.aws_connector = AWSConnector()

    def upload_to_s3(self, folder):

        try:
            self.aws_connector.send_to_s3(folder)
            print("Sent to S3")
            return True
        except Exception as e:
            print(e)
            return False