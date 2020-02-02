import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings; warnings.simplefilter('ignore')
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

data = pd.read_csv("diabetes.csv")

X = data.drop(['Outcome'], axis=1)
y = data.Outcome

X = (X - X.min())/(X.max()-X.min())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


pickle.dump(lr, open('dmodel.pkl','wb'))

dmodel = pickle.load(open('dmodel.pkl','rb'))



