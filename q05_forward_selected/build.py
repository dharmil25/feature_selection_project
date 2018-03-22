# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import r2_score
from operator import itemgetter

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here
def forward_selected(data,model):
    features, target = data.iloc[:,:-1], data.iloc[:,-1]
    selected_features =list()
    selected_r2_scores =list()
    lst = []
    for i in range(1,len(features.columns)):
        for colname in features:
            iteration_features =selected_features[:]
            iteration_features.append(colname)
            model.fit(features.loc[:,iteration_features],target)
            r = r2_score(target, model.predict(features.loc[:,iteration_features]))
            lst.append([colname,r])
        scores_iterations = sorted(lst, key=itemgetter(1),reverse=True)
        selected_features.append( scores_iterations[0][0])
        selected_r2_scores.append( scores_iterations[0][1])
    return selected_features, selected_r2_scores
