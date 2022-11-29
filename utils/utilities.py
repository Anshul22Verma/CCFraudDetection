import boto3
from botocore.exceptions import NoCredentialsError
import json
import os.path as osp

class InvalidInputException(Exception):
    pass

class InternalException(Exception):
    pass


ACCESS_KEY = 'XXXXXXXXXXXXXXXXXXXXXXX'
SECRET_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
googleAuth = 'XXXXXXXXXXXXXXXX'

with open(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'cred.json')) as data_file:
    data = json.load(data_file)
    ACCESS_KEY = data['ACCESS_KEY']
    SECRET_KEY = data['SECRET_KEY']
    googleAuth = data['googleAuth']


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def download_from_aws(bucket, s3_file, local_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.download_file(bucket, s3_file, local_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def send_email() -> None:
    import smtplib
    from email.message import EmailMessage

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    #This is where you would replace your password with the app password
    server.login('iitm.anshul.verma@gmail.com', googleAuth)

    msg = EmailMessage()

    message = 'please check `test.csv`, add labels to it and test the model predictions to ensure that the model is not stale.\nIf its stale use the newer data for re-training.'
    msg.set_content(message)
    msg['Subject'] = "cc Fraud stale check"
    msg['From'] = 'ccFraudDemo@gmail.com'
    msg['To'] = 'av.vermaans@gmail.com'
    server.send_message(msg)
