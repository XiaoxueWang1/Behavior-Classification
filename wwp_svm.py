# -*- coding: utf-8 -*-
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score

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
    
data = pd.read_csv("train.csv")

# delete some bad data
remove_temp = data.drop(['Category'], 1)
remove_index = outlier_hunt(remove_temp)
data = data.drop(remove_index, axis=0)
data = data.reset_index(drop=True)

'''
data = data[data['Attribute6']<0]
data = data.reset_index(drop=True)
'''

X_data = data.drop(['Category'], 1)
Y_data = data['Category']

test_data = pd.read_csv("test.csv")


X_data['Attribute6'] = X_data['Attribute6'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute6'] = test_data['Attribute6'].map(lambda x: x/abs(x)*np.log(abs(x)))
'''
X_data['Attribute6'] = X_data['Attribute6'].map(lambda x: np.log(abs(x)))
test_data['Attribute6'] = test_data['Attribute6'].map(lambda x: np.log(abs(x)))
'''

X_data['Attribute4'] = X_data['Attribute4'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute4'] = test_data['Attribute4'].map(lambda x: x/abs(x)*np.log(abs(x)))

X_data['Attribute1'] = X_data['Attribute1'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute1'] = test_data['Attribute1'].map(lambda x: x/abs(x)*np.log(abs(x)))


X_data['Attribute5'] = X_data['Attribute5'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute5'] = test_data['Attribute5'].map(lambda x: x/abs(x)*np.log(abs(x)))

X_data['Attribute2'] = X_data['Attribute2'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute2'] = test_data['Attribute2'].map(lambda x: x/abs(x)*np.log(abs(x)))

T = X_data.append(test_data)
scaler = StandardScaler()
scaler.fit(T)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_data = scaler.transform(X_data)
test_data = scaler.transform(test_data)

#X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0)
X_train = X_data
Y_train = Y_data
Y_train = np.asarray(Y_train)

clf = svm.SVC(C=40,gamma=7.5)
clf.fit(X_train,Y_train)
pre = clf.predict(test_data)
ts = pd.DataFrame(pre)
ts = pd.DataFrame({"Id":range(1,len(ts)+1),"Category":pre})
cols = list(ts)
cols.insert(0, cols.pop(cols.index('Id')))
ts = ts.ix[:, cols]
ts.to_csv("11849331-submission.csv",index = False,header=True)