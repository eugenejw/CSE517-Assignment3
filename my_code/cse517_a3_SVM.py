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
from sklearn.multiclass import OneVsOneClassifier
import operator
import numpy as np
from sklearn.metrics import accuracy_score

author_dic = {'CLINTON_PRIMARY2008\n': 3, 'ROMNEY_PRIMARY2008\n': 4, 'PAUL_PRIMARY2012\n': 1, 'HUNTSMAN_PRIMARY2012\n': 16, 'SANTORUM_PRIMARY2012\n': 13, 'ROMNEY_PRIMARY2012\n': 12, 'HUCKABEE_PRIMARY2008\n': 11, 'EDWARDS_PRIMARY2008\n': 8, 'BIDEN_PRIMARY2008\n': 17, 'OBAMA_PRIMARY2008\n': 0, 'CAIN_PRIMARY2012\n': 18, 'RICHARDSON_PRIMARY2008\n': 7, 'THOMPSON_PRIMARY2008\n': 10, 'GIULIANI_PRIMARY2008\n': 9, 'BACHMANN_PRIMARY2012\n': 5, 'MCCAIN_PRIMARY2008\n': 2, 'PAWLENTY_PRIMARY2012\n': 15, 'PERRY_PRIMARY2012\n': 14, 'GINGRICH_PRIMARY2012\n': 6}

df_dev = pd.read_csv('/Volumes/HD/working_directory/CSE517/assignment3/my_code/dev_set.csv',sep=',')
df = pd.read_csv('/Volumes/HD/working_directory/CSE517/assignment3/my_code/train_set.csv',sep=',')

# Assuming same lines from your example
cols_to_norm = ['ave_word_freq','#_of_numbers', 'ave_2gram_nor', 'ave_3gram_nor']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
df_dev[cols_to_norm] = df_dev[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))


df_data = df.ix[:, 0:24]
df_target = df.ix[:,-1:]
df_dev_data = df_dev.ix[:, 0:24]
df_dev_target = df_dev.ix[:,-1:]
X, y = df_data, df_target

clf = OneVsRestClassifier(SVC(random_state=0,probability=True)).fit(X, y)
#predict the dev set
predicted = clf.predict(df_dev_data)
predicted = list(predicted)
true = df_dev_target['numeric_author'].tolist()
accuracy_score(true, predicted)

#acc for dev set
df_unlabled = pd.read_csv('/Volumes/HD/working_directory/CSE517/assignment3/my_code/unlabeled_set.csv',sep=',')
df_unlabled[cols_to_norm] = df_unlabled[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
df_unlabled_org = df_unlabled
df_unlabled = df_unlabled.drop('filename', 1)
#df_unlabled = df_unlabled.drop('numeric_author', 1)
#predict the unlabled
predicted = clf.predict(df_unlabled)
proba_matrix = clf.predict_proba(df_unlabled)
matrix_lst = proba_matrix.tolist()


count = 0
local_max = []
for lst in matrix_lst:
    max_index, max_value = max(enumerate(lst), key=operator.itemgetter(1))
    local_max.append((max_value, count))
    count += 1
    
local_max.sort(reverse=True)

top1 = local_max[: int(len(local_max) * 0.1)]

df_removed = df_unlabled
index_to_be_droped = [x[1] for x in top1]
index_to_be_droped_org = index_to_be_droped[:]
ret_authors = []
for i in index_to_be_droped:
    num_au = int(predicted[i])
    author = ''
    for au in author_dic:
        if author_dic[au] == num_au:
            author = au
    ret_authors.append(author)

file_list = []
with open('/Volumes/HD/working_directory/CSE517/assignment3/my_code/unlabeled.tsv', 'r') as f:
    counter = 0
    for line in f:
        print line
        if counter in index_to_be_droped:
            #print 'here'
            line = line.strip()
            line = line[3:]
            file_list.append(line)
            index_to_be_droped.remove(counter)
        counter += 1

ret_lst = zip(file_list, ret_authors)
with open('/Volumes/HD/working_directory/CSE517/assignment3/my_code/top1.tsv', 'w') as f:
    for each in ret_lst:
        s = '\t'.join((each[0], each[1]))
        f.write(s)
        
#concat files
filenames = ['/Volumes/HD/working_directory/CSE517/assignment3/my_code/train.tsv', '/Volumes/HD/working_directory/CSE517/assignment3/my_code/top1.tsv']
with open('/Volumes/HD/working_directory/CSE517/assignment3/my_code/train_iter1.tsv', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

#remove lines from unlabeled
with open('/Volumes/HD/working_directory/CSE517/assignment3/my_code/unlabeled.tsv', 'r') as f:
    with open('/Volumes/HD/working_directory/CSE517/assignment3/my_code/unlabeled_iter1.tsv', 'w') as outfile:
        counter = 0
        for line in f:
            if counter in index_to_be_droped_org:
            #print 'here'
                index_to_be_droped_org.remove(counter)
                counter += 1
            else:
                outfile.write(line)
                counter += 1
    





df_removed = df_removed.drop(df_removed.index[index_to_be_droped])

df.drop(df.index[[1,3]])

df_unlabled = df_removed
















df_data = df.ix[:, 0:24]
df_target = df.ix[:,-1:]
df_dev_data = df_dev.ix[:, 0:24]
df_dev_target = df_dev.ix[:,-1:]
X, y = df_data, df_target

#lr
#logreg = linear_model.LogisticRegression(C=1e5)
#logreg.fit(X, y)
#logreg.predict(X)

df_unlabled = pd.read_csv('/Volumes/HD/working_directory/CSE517/assignment3/my_code/unlabeled_set.csv',sep=',')

#one Vs Rest
#clf_0 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
clf = OneVsRestClassifier(SVC(random_state=0,probability=True)).fit(X, y)

#predict unlabeled set
predicted = clf.predict(df_unlabled)
proba_matrix = clf.predict_proba(df_unlabled)
matrix_lst = proba_matrix.tolist()
count = 0
local_max = []
for lst in matrix_lst:
    max_index, max_value = max(enumerate(lst), key=operator.itemgetter(1))
    local_max.append((max_value, count)
    count += 1
    

#predict the dev set
predicted = clf.predict(df_dev_data)
predicted = list(predicted)
true = df_dev_target['numeric_author'].tolist()


import numpy as np
from sklearn.metrics import accuracy_score
accuracy_score(true, predicted)
#predicted = clf.predict(X)
#true = df_target['numeric_author'].tolist()
#predicted == true



df = df.drop('target', 1)