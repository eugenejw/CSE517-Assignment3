from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

OneVsRestClassifier(SVC(random_state=0,probability=True)).fit(X, y).predict(X)
OneVsRestClassifier(SVC(random_state=0,probability=True )).fit(X, y).predict_proba(X)


import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import tree
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

df_dev = pd.read_csv('/Volumes/HD/working_directory/CSE517/assignment3/my_code/dev_set.csv',sep=',')
df = pd.read_csv('/Volumes/HD/working_directory/CSE517/assignment3/my_code/train_set.csv',sep=',')

df_data = df.ix[:, 0:24]
df_target = df.ix[:,-1:]
X, y = df_data, df_target
clf = OneVsRestClassifier(SVC(random_state=0,probability=True)).fit(X, y)

df_dev_data = df_dev.ix[:, 0:24]


#predicted = clf.predict(X)
#true = df_target['numeric_author'].tolist()
#predicted == true



df = df.drop('target', 1)
