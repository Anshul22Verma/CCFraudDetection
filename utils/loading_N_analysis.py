import matplotlib.pyplot as plt
import missingno
import os
import os.path as osp
import pandas as pd
import seaborn as sns 
import sys

sys.path.append('/Users/anshulverma/Documents/CCFraudDetection')
from utils.utilities import InvalidInputException


class CCFraudData:
    def __init__(self, loc: str, target_loc: str) -> None:
        self.loc = loc
        self.df = self.load_data(loc=self.loc)
        self.target_col = 'Class'
        self.target_loc = target_loc
    
    @staticmethod
    def load_data(loc: str) -> pd.DataFrame:
        '''
            loads the csv data
        '''
        df = pd.read_csv(loc)
        return df
    
    def analyze(self, df: pd.DataFrame = None) -> None:
        '''
            analyze the dataset
        '''
        if not df:
            if not self.df.empty:
                df = self.df
            else:
                return InvalidInputException(f'df and self.df are both None')

        if self.target_col not in df.columns:
            return InvalidInputException(f'{self.target_col} not in df')

        
        fig = self.nan_frequency(df=df, target_col=self.target_col)
        fig.savefig(osp.join(self.target_loc, 'nan_dist.png'))
        plt.close(fig)

        fig = self.val_spread_plot(df=df)
        fig.savefig(osp.join(self.target_loc, 'value_spread.png'))
        plt.close(fig)

        fig = self.corr_plot(df=df, target_col=self.target_col)
        fig.savefig(osp.join(self.target_loc, 'corr.png'))
        plt.close(fig)

        fig = self.box_plot(df=df, target_col=self.target_col)
        fig.savefig(osp.join(self.target_loc, 'box.png'))
        plt.close(fig)
        
        fig = self.class_distribution(df=df, target_col=self.target_col)
        fig.savefig(osp.join(self.target_loc, 'distribution.png'))
        plt.close(fig)

    @staticmethod
    def nan_frequency(df: pd.DataFrame, target_col: str) -> plt.figure:
        '''
            Plot showing nan values for different columns at different rows
        '''
        fig = plt.figure(figsize=(12, 15))
        ax = plt.gca()
        missingno.matrix(df, labels=True, ax=ax, sparkline=False)
        fig.suptitle("NaN disribution w/ white corresponding to NaN")
        return fig
        
    @staticmethod
    def val_spread_plot(df: pd.DataFrame) -> plt.figure:
        '''
            Returns a plot showing value distribution of all the columns (to check normalization requirement) 
        '''
        fig = plt.figure(figsize=(35, 15))
        axs = fig.subplots(nrows=int(len(df.columns)/8)+1, ncols=8)
        for idx, col in enumerate(df.columns):
            axs[int(idx/8)][idx%8].hist(df[col].values.tolist())
            axs[int(idx/8)][idx%8].set_title(f'val distribution of {col} column')
        fig.suptitle("Value spread of all the columns")
        return fig

    @staticmethod
    def corr_plot(df: pd.DataFrame, target_col: str) -> plt.figure:
        '''
            Returns correlation plot (to see feature relevance) of all the features agaist the target column
        '''
        fig = plt.figure(figsize=(14,11))
        ax = plt.gca()
        corr = df.corr()[target_col].sort_values().drop(target_col)
        corr.plot(kind='barh', ax=ax)
        ax.grid(True)
        fig.suptitle("Correlation between Features")
        return fig

    @staticmethod
    def box_plot(df: pd.DataFrame, target_col: str) -> plt.figure:
        '''
            Creates a box plot (to see feature relevance) of all the features against the target column
        '''
        fig = plt.figure(figsize=(24, 15))
        axs = fig.subplots(nrows=int(len(df.columns)/4)+1, ncols=4)
        for idx, feature in enumerate(df.columns):
            if feature != target_col:
                idx += 1
                sns.boxplot(y=feature, x=target_col, data=df, ax=axs[int(idx/4)][idx%4])
        fig.suptitle(f'Box plot of each column against {target_col}')
        return fig

    @staticmethod
    def class_distribution(df: pd.DataFrame, target_col: str) -> plt.figure:
        '''
            Creates a plot to visualize class distribution
        '''
        fig = plt.figure(figsize=(24, 15))
        axs = fig.subplots(nrows=1, ncols=2)

        # plotting a pie chart in axes 0
        explode = [0, 0.1]
        axs[0].pie(df[target_col].value_counts(), explode=explode, autopct='%1.1f%%', shadow=True, startangle=140)
        axs[0].legend(labels=['Normal','Fraud'])
        axs[0].set_title('"Fraud" Distribution, Pie')
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        
        # plotting a bar chart in axes 1
        sns.countplot(x=target_col, data=df, ax=axs[1])
        axs[1].set_title('"Fraud" Distribution, Bar')
        fig.suptitle(f'Unbalanced class "{target_col}" distribution')
        return fig

               
if __name__ == "__main__":
    loc = '/Users/anshulverma/Documents/CCData/creditcard.csv'
    ccData = CCFraudData(loc=loc, target_loc='/Users/anshulverma/Documents/CCData/analysis')
    ccData.analyze()
