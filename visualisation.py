import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def countplot(df,feature):
    fig = plt.plot
    sns.countplot(x=df[feature])

def corr_map(df):
    corr_mtrx=df.corr()
    corr_mtrx
    # draw the heatmap 
    plt.figure(figsize = (14,12))
    ax = sns.heatmap(corr_mtrx, linewidths=.5, annot=True, cmap='coolwarm')