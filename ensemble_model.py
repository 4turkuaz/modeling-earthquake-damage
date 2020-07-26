import os
import sklearn
import pandas as pd
import xgboost as xgb
import numpy as np
import lightgbm as lgbm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

TRAIN_VALUES = 'DATA/train_values.csv'
TRAIN_LABELS = 'DATA/train_labels.csv'
TEST_VALUES = 'DATA/test_values.csv'

clf1 = GaussianProcessClassifier()
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = MLPClassifier()
clf5 = KNeighborsClassifier(7)
clf6 = SVC(kernel = 'linear')
clf7 = SVC(kernel = 'rbf')
clf8 = SVC(kernel = 'poly')
clf9 = AdaBoostClassifier()
clf10 = xgb.XGBClassifier()
clf11 = QuadraticDiscriminantAnalysis()
clf12 = lgbm.LGBMClassifier()

classifiers = [
	#('gaus', clf1),
	#('rf', clf2),
	#('gnb', clf3),
	#('mlp', clf4),
	#('knn', clf5),
	#('svc_linear', clf6),
	#('svc_rbf', clf7),
	#('poly', clf8),
	('adaboost', clf9),
	('xgb', clf10),
	#('quad', clf11),
	('lgbm', clf12)
]

eclf1 = VotingClassifier(estimators=classifiers, voting='soft', n_jobs = -1)


train_x = pd.read_csv(TRAIN_VALUES)
train_y = pd.read_csv(TRAIN_LABELS)
test_x = pd.read_csv(TEST_VALUES)

from sklearn.metrics import f1_score

results = []
print('Time started: ' + datetime.now().strftime("%H:%M:%S"))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(train_x.values[:, 1:]):
    eclf1 = eclf1.fit(train_x.values[train_index, 1:], train_y.values[train_index, 1])
    preds = eclf1.score(train_x.values[test_index, 1:], train_y.values[test_index, 1])
    f1_score_val = f1_score(preds, train_y.values[test_index, 1])
    results.append(f1_score_val)

print(results)
print(np.mean(results))
print('Time end: ' + datetime.now().strftime("%H:%M:%S"))
