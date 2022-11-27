import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix

import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


def baseline(df, feature, threshold):
    '''Baseline Model. Separates dataframe by only one feature.
    If the feature is below the threshold, the output is 1, if
    the feature is above the threshold, the output is 0.
    '''
    y_pred = [1 if x < threshold else 0 for x in df[feature]] 
    return y_pred

def eval_metrics(y_true, y_pred, model="current model"):
    '''Simple evaluation function. Prints accuracy, recall and
    precision as values and plots a confusion matrix.
    '''
    print(model,":")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, cmap="YlGnBu_r", annot=True, fmt=".0f");

def simple_model(X_train, y_train, X_test):
    '''Both fitting and prediction for a simple logistic regression.
    (Not as simple as the baseline model though.)
    '''
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred