import boto3
from fastapi import Request,FastAPI
import os
import os.path as osp
import pandas as pd
from pydantic import BaseModel
import uvicorn
import sys

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from exp.model.ml_models import MLClassifierModel, load_model
from utils.utilities import download_from_aws, upload_to_aws, send_email


app = FastAPI()

# dowload model from s3 bucket
curr = os.getcwd()
tmp_dir = osp.join(curr, 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
model = MLClassifierModel((1, 1))
if download_from_aws('ccfraudbucket', 'raw/model.json', osp.join(tmp_dir, 'model.json')):
    model = load_model(osp.join(tmp_dir, 'model.json'))
else:
    Exception("can't download model")


class FraudRequest(BaseModel):
    Time: int
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


def get_predictions(request: dict, model: MLClassifierModel):
    df = pd.DataFrame()
    df.append(request)
    result = model.best_model.best_estimator_.predict(df)
    return {'prediction_class': result}, df


@app.get('/')
async def home():
    return {"message": "SGS & Co Demo"}

@app.post("/prediction")
async def getsummary(user_request_in: FraudRequest):
    payload = {'Time': user_request_in.time, 'Amount': user_request_in.amount,
        'V1': user_request_in.V1, 'V2': user_request_in.V2, 'V3': user_request_in.V3, 'V4': user_request_in.V4,
        'V5': user_request_in.V4, 'V6': user_request_in.V6, 'V7': user_request_in.V7, 'V8': user_request_in.V8, 
        'V9': user_request_in.V9, 'V10': user_request_in.V10, 'V11': user_request_in.V11, 'V12': user_request_in.V12,
        'V13': user_request_in.V13, 'V14': user_request_in.V14, 'V15': user_request_in.V15, 'V16': user_request_in.V16,
        'V17': user_request_in.V17, 'V18': user_request_in.V18, 'V19': user_request_in.V19, 'V20': user_request_in.V20,
        'V21': user_request_in.V21, 'V22': user_request_in.V22, 'V23': user_request_in.V23, 'V24': user_request_in.V24,
        'V25': user_request_in.V25, 'V26': user_request_in.V26, 'V27': user_request_in.V27, 'V28': user_request_in.V28,
        }
    
    response, df_new = get_predictions(payload, model)
    # to ensure the device using which the prediction was made
    if download_from_aws('ccfraudbucket', 'test.csv', osp.join(tmp_dir, 'test.csv')):
        df = pd.read_csv(osp.join(tmp_dir, 'test.csv'))
        df.concat(df_new)
        if len(df) > 5:
            # send user an email asking to annotate new data and evaluate performance to ensure model is not stale
            send_email()
        df.to_csv(osp.join(tmp_dir, 'test.csv'))
        upload_to_aws('ccfraudbucket', 'test.csv', osp.join(tmp_dir, 'test.csv'))
    else:
        df_new.to_csv(osp.join(tmp_dir, 'test.csv'))
        upload_to_aws('ccfraudbucket', 'test.csv', osp.join(tmp_dir, 'test.csv'))
    return response