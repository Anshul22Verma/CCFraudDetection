import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

class BestModel:
    def __init__(self, model: str, cross_val_score: float, best_estimator_) -> None:
        self.model = model
        self.cross_val_score = cross_val_score
        self.best_estimator_ = best_estimator_


class MLClassifierModel:
    '''
        The class performs grid-search for different ML models and returns the best set of parameters.
    '''
    def __init__(self, train: tuple, scaler: bool = False) -> None:
        self.X, self.y = train

        if scaler:
            _scaler = StandardScaler().fit(self.X)
            self.X = _scaler.transform(self.X)
        
        self.auc_n_models = []
        # to identify the type of best model
        self.best_model = None

        # default ADA classifier
        DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced")
        # different classificartion models to search from
        self.models = {
                'LR1': LogisticRegression(), 'LR2': LogisticRegression(), 'LR3': LogisticRegression(), 
                'SGD': SGDClassifier(), 'Ridge': RidgeClassifier(), 'RF': RandomForestClassifier(),
                'AdaBoost': AdaBoostClassifier(base_estimator=DTC), 'GB': GradientBoostingClassifier(),
                'ExT': ExtraTreesClassifier(), 'DT': DecisionTreeClassifier()
            }
        self.params = {
                'LR1': {
                    'penalty': ['l2'], 'C': np.linspace(0.1, 100, 10),
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']
                    },
                'LR2': {
                    'penalty': ['l1'], 'C': np.linspace(0.1, 1000, 10),
                    'solver': ['saga']
                    },
                'LR3': {
                    'penalty': ['elasticnet'], 'C': np.linspace(0.1, 1000, 10),
                    'l1_ratio': np.linspace(0.1, 1, 10), 'solver': ['saga']
                    },
                'SGD': {
                    'alpha': np.linspace(1e-5, 100, 10), 
                    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                    'penalty': ['l2', 'l1', 'elasticnet']
                    },
                'Ridge': {'alpha': np.linspace(1e-4, 100, 10)},
                'RF': {
                    'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth' : [5, 10, 15, 20], 'criterion' :['gini', 'entropy']
                    },
                'AdaBoost': {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [200, 500]
                    },
                'GB': {
                    'loss': ['deviance'],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [5, 10, 15, 20],
                    'n_estimators': [200, 500]
                    },
                'ExT': {
                    'max_depth': [5, 10, 15, 20], 'criterion' :['gini', 'entropy'],
                    },
                'DT': {
                    'max_depth' : [5, 10, 15, 20], 'criterion' :['gini', 'entropy']
                    }
                }

    def fit(self) -> None:
        '''
            Performs Grid Search for different models and reports the best model.
        '''
        print('#'*10 + ' '*5 + 'PERFORMING EXTENSIVE GRID-SEARCH' + ' '*5 + '#'*10)

        for model in self.models.keys():
            gs = GridSearchCV(estimator=self.models[model], param_grid=self.params[model], refit=True, scoring='roc_auc', cv=3)
            _ = gs.fit(self.X, self.y)
            print(f'VALIDATION SCORE of {model}:', gs.best_score_)
            self.auc_n_models.append(BestModel(model=model, cross_val_score=gs.best_score_, best_estimator_=gs.best_estimator_))
            
        print('#'*10 + ' '*5 + '-'*20 + ' '*5 + '#'*10)
        print('\n\n')
        self.select_best_model()

    def select_best_model(self) -> None:
        '''
            returns the best model's name and its validation score (roc-auc-score).
        '''
        sorted_scored = sorted(self.auc_n_models, key=lambda x: (x.cross_val_score), reverse=True)
        self.best_model = sorted_scored[0]
        print('Selected best model is {}, 3 fold cross-validation score of this model is {}.'.format(self.best_model.model, self.best_model.cross_val_score))

    
    def plot_best_model(self, X: pd.DataFrame, y: list, title: str = ''):
        '''
            Makes the plot for the best model
        '''
        if X is None or y is None or len(X) != len(y):
            Exception('Need both X and y and they must have the same length')
        
        cls = self.best_model.best_estimator_
        y_pred_proba, y_pred = cls.predict_proba(X)[:, 1], cls.predict(X)
        
        auc = roc_auc_score(y_true=y, y_score=y_pred_proba, average='weighted')
        acc = accuracy_score(y_true=y, y_pred=y_pred)
        fpr, tpr, _ = roc_curve(y,  y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        
        # plotting
        fig = plt.figure()
        ax = fig.subplots()
        lw = 2 
        ax.plot(fpr, tpr, color="darkorange", 
            lw=lw, label="ROC curve (area = %0.2f)" % auc)
        ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend()
        fig.suptitle(title + f' Accuracy={round(acc, 4)}, AUC={round(auc, 4)}')
        return fig

    def roc_auc_score_(self, X: pd.DataFrame, y: list):
        '''
            returns ROC-AUC score on the dataset
        '''
        if X is None or y is None or len(X) != len(y):
            Exception('Need both X and y and they must have the same length')
        
        cls = self.best_model.best_estimator_
        return roc_auc_score(y, cls.predict_proba(X)[:, 1])

import pickle

def save_model(obj: MLClassifierModel, full_path: str) -> None:
    '''
        save the model in a `pickle` file
    '''
    #save it
    with open(full_path, 'wb') as f:
        pickle.dump(obj, f) 


def load_model(full_path: str) -> MLClassifierModel:
    '''
        load the model from a `pickle` file
    '''
    #load it
    with open(full_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
    

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/anshulverma/Documents/CCFraudDetection')
    
    from utils.loading_N_analysis import CCFraudData
    from utils.preprocess import CCFraudPreprocess

    loc = '/Users/anshulverma/Documents/CCData/creditcard.csv'
    ccData = CCFraudData(loc=loc, target_loc='/Users/anshulverma/Documents/CCData/analysis')
    ccPreprocess = CCFraudPreprocess(ccData=ccData)

    train_df, test_df = ccPreprocess.split_dataframe()
    X_train, y_train = ccPreprocess.get_XnY(train_df)
    # perform upsampling
    X_train, y_train = ccPreprocess.down_sample(X_df=X_train, y_col=y_train)
    train= (X_train, y_train)

    # classifier
    cls_ = MLClassifierModel(train=train, scaler=False)
    cls_.fit()
