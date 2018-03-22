# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    X,y = df.iloc[:,:-1],df.iloc[:,-1]
    clf = RandomForestClassifier(random_state=9)
    model = SelectFromModel(clf)
    model.fit_transform(X,y)
    return X.columns.values[model.get_support()].tolist()
