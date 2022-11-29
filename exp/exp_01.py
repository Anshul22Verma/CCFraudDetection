import matplotlib.pyplot as plt
import os.path as osp
import pickle
import sys
sys.path.append('/Users/anshulverma/Documents/CCFraudDetection')

from utils.loading_N_analysis import CCFraudData
from utils.preprocess import CCFraudPreprocess
from exp.model.ml_models import MLClassifierModel, load_model, save_model



if __name__ == "__main__":
    loc = '/Users/anshulverma/Documents/CCData/creditcard.csv'
    target_loc='/Users/anshulverma/Documents/CCData/analysis'
    ccData = CCFraudData(loc=loc, target_loc=target_loc)
    # ccData.analyze()
    ccPreprocess = CCFraudPreprocess(ccData=ccData)

    train_df, test_df = ccPreprocess.split_dataframe()
    X_train, y_train = ccPreprocess.get_XnY(train_df)
    # perform upsampling
    X_train, y_train = ccPreprocess.down_sample(X_df=X_train, y_col=y_train)
    train = (X_train, y_train)

    # classifier
    cls_ = MLClassifierModel(train=train, scaler=False)
    cls_.fit()

    save_model(cls_, osp.join(target_loc, 'model.json'))

    cls_ = load_model(osp.join(target_loc, 'model.json'))
    print(f'Cross Validation training ROC AUC score of the best estimator is {round(cls_.best_model.cross_val_score, 4)}')
    fig = cls_.plot_best_model(X=X_train, y=y_train, title='ROC Curve on training set')
    fig.savefig(osp.join(target_loc, 'training_ROC.png'))
    plt.close(fig)

    # testing
    X_test, y_test = ccPreprocess.get_XnY(test_df)
    test_score = cls_.roc_auc_score_(X=X_test, y=y_test)
    print(f'Test ROC AUC score of the best estimator is {round(test_score, 4)}')
    fig = cls_.plot_best_model(X=X_test, y=y_test, title='ROC Curve on test set')
    fig.savefig(osp.join(target_loc, 'test_ROC.png'))
    plt.close(fig)
    