# CCFraudDetection

Credit Card fraud detection to detect CC fraud via PCA features to hide the details.

## Setup
Create a `CCData` directory where you clone the repository and put the csv data file inside that folder. All the results are then locally saved in an analysis folder inside the Data folder.


## Dataset
Using kaggle dataset to build a model which detects fraudulent transaction 
ref: [CCFraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

## Model
#### Algorithm
Used a big grid of different ML algorithms and hyperparameters to select an algorithm and model.


#### Performance
```
##########     PERFORMING EXTENSIVE GRID-SEARCH     ##########
VALIDATION SCORE of LR1: 0.9836443586443587
VALIDATION SCORE of LR2: 0.6126659270964291
VALIDATION SCORE of LR3: 0.6126813520635914
VALIDATION SCORE of SGD: 0.6272421069718367
VALIDATION SCORE of Ridge: 0.9734693670504481
VALIDATION SCORE of RF: 0.9825630063323114
VALIDATION SCORE of AdaBoost: 0.9061561561561562
VALIDATION SCORE of GB: 0.9814862090441241
VALIDATION SCORE of ExT: 0.9838414669495751
VALIDATION SCORE of DT: 0.9326985389968016
##########     --------------------     ##########
```

## WorkFlow
- Train a model and deploy it in the cloud with versioning.
- Predection requests > 5 sends a notification to the user, with a request to label the new data and check if the model is getting stale.

