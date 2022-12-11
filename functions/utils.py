import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def pivot(df, column, by='cluster', agg='count'):
    pivot_table = pd.pivot_table(df, 
               columns = column, 
               values = df.columns[0],
               index = by, 
               fill_value = 0,
               aggfunc = agg,
               observed = True) 
    pivot_table = round(pivot_table.div(pivot_table.sum(axis = 1), axis=0), 3) * 100
    pivot_table.rename(
        columns= {col:str(col)+'%' for col in pivot_table.columns}, inplace=True)
    return pivot_table


def display_clusters(X, y=None, means=None):
    if y is None:
        y = 'grey'
    plt.scatter(X[:, 0], X[:, 1], c=y)
    if means is not None:
        m = np.vstack(means)
        plt.scatter(m[:,0], m[:,1], c=range(m.shape[0]), marker='^')
    plt.rcParams['figure.figsize'] = [3, 3]
    plt.show()