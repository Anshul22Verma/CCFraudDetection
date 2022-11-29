from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
import sys

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.loading_N_analysis import CCFraudData
from utils.utilities import InternalException


class CCFraudPreprocess:
    def __init__(self, ccData: CCFraudData) -> None:
        self.df = ccData.df
        self.target_col = ccData.target_col
        # not setting a random state because we want to get different models everytime ONLY used in the test split TO NEVER see the TEST set
        self.random_ = 2022

    def split_dataframe(self, test_r: float = 0.1, stratify: bool = True) -> tuple:
        '''
            Splits the data into training, validation and testing sets for ML usage -> (train_df, val_df, test_df)
        '''
        if test_r <= 0 or test_r >= 1:
            return InternalException('Can not have a ratio outside range [0, 1]')

        stratify_col = None
        if stratify:
            stratify_col = self.df[self.target_col].values.tolist()
        train_df, test_df = train_test_split(self.df, test_size=test_r, stratify=stratify_col, random_state=self.random_)

        return (train_df, test_df)

    def get_XnY(self, df: pd.DataFrame) -> tuple:
        '''
            turns df -> (X, y) by dropping the target column and returning it seperately
        '''
        return (df.drop(columns=self.target_col, axis=1), df[self.target_col].values.tolist())
    
    @staticmethod
    def up_sample(X_df: pd.DataFrame, y_col: list) -> tuple:
        ros = RandomOverSampler()
        X_df, y_col = ros.fit_resample(X_df, y_col)
        return (X_df, y_col)
    
    @staticmethod
    def up_SMOTE(X_df: pd.DataFrame, y_col: list) -> tuple:
        X_df, y_col = SMOTE().fit_resample(X_df, y_col)
        return (X_df, y_col)

    @staticmethod
    def down_sample(X_df: pd.DataFrame, y_col: list) -> pd.DataFrame:
        rus = RandomUnderSampler()
        X_df, y_col = rus.fit_resample(X_df, y_col)
        return (X_df, y_col)

    
if __name__ == "__main__":  
    main_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    os.makedirs(osp.join(main_dir, 'CCData'), exist_ok=True)
    loc = osp.join(main_dir, 'CCData', 'creditcard.csv')
    target_loc = osp.join(main_dir, 'CCData', 'analysis')
    os.makedirs(target_loc, exist_ok=True)

    ccData = CCFraudData(loc=loc, target_loc=target_loc)

    print('#'*10 + ' '*5 + 'Preprocessing FRAUD datasets' + ' '*5 + '#'*10)
    ccPreprocess = CCFraudPreprocess(ccData=ccData)
    # split the dataframe
    train_df, test_df = ccPreprocess.split_dataframe()

    X_train, y_train = ccPreprocess.get_XnY(train_df)
    # before balancing the class distribution in the training set is
    cls, counts = np.unique(y_train, return_counts=True)
    print(f'Before balancing the number of unique classes and their values in the training set are CLS: {cls}, COUNT: {counts}')

    # to deal with data imbalance lets try to upsample the dataset
    X_train_, y_train_ = ccPreprocess.up_sample(X_df=X_train, y_col=y_train)
    cls, counts = np.unique(y_train_, return_counts=True)
    print(f'After upsampling, # of samples in classes in training set are CLS: {cls}, COUNT: {counts}')
    # Just saving one image after upsampling for display
    X_train_[ccData.target_col] = y_train_
    fig = ccData.class_distribution(df=X_train_, target_col=ccData.target_col)
    fig.savefig(osp.join(ccData.target_loc, 'distribution_postOverSampling.png'))
    plt.close(fig)
    
    # to deal with data imbalance lets try to upsample the dataset via SMOTE 
    X_train_, y_train_ = ccPreprocess.up_SMOTE(X_df=X_train, y_col=y_train)
    cls, counts = np.unique(y_train_, return_counts=True)
    print(f'After SMOTE upsampling, # of samples in classes in training set are CLS: {cls}, COUNT: {counts}')

    # to deal with data imbalance lets try to downsample the dataset
    X_train_, y_train_ = ccPreprocess.down_sample(X_df=X_train, y_col=y_train)
    cls, counts = np.unique(y_train_, return_counts=True)
    print(f'After downsampling, # of samples in classes in training set are CLS: {cls}, COUNT: {counts}')
