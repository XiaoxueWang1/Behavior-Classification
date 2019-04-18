import pandas as pd
import numpy as np
from numpy.random import seed
#from collections import Counter
from sklearn import tree
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
'''
def outlier_hunt(df):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than 2 outliers.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in df.columns.tolist():
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 1)

        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 99)

        # Interquartile rrange (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v >= 2)

    return multiple_outliers
    
# delete some bad data
remove_temp = data.drop(['Category'], 1)
remove_index = outlier_hunt(remove_temp)
data = data.drop(remove_index, axis=0)
data = data.reset_index(drop=True)

data = data[data['Attribute6'] < 0]
data = data.reset_index(drop=True)
'''
X = data.drop(['Attribute6', 'Category'], 1)
Y = data['Category']
test_data = test_data.drop(['Attribute6'], 1)

seed(9078)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)
Y_test_predicted = clf.predict(test_data)
ts = pd.DataFrame(Y_test_predicted)
ts = pd.DataFrame({"Id":range(1,len(ts)+1),"Category":Y_test_predicted})
cols = list(ts)
cols.insert(0, cols.pop(cols.index('Id')))
ts = ts.loc[:, cols]
#ts = ts.ix[:, cols]
ts.to_csv("11849332-submission.csv",index = False,header=True)