# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    sp = SelectPercentile(f_regression,percentile=k)
    sp.fit_transform(X,y)
    features = X.columns.values[sp.get_support()]
    scores = sp.scores_[sp.get_support()]
    fs_score = list(zip(features,scores))
    df = pd.DataFrame(fs_score,columns=['Name','Score'])
    return df.sort_values(['Score','Name'],ascending = [False,True])['Name'].tolist()
