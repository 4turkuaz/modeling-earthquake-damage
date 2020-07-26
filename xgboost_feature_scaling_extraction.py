import os
import sklearn
import pandas as pd
import xgboost as xgb
import numpy as np
from numpy import sort
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.datasets import load_iris, load_digits, load_boston
from datetime import datetime
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train_x = pd.read_csv('DATA/train_values.csv')
train_y = pd.read_csv('DATA/train_labels.csv')
test_x = pd.read_csv('DATA/test_values.csv')

# print(train_y.shape)

# print(train_x[0, 8:15])
categorical_columns = list(train_x.columns[8:15]) + ['legal_ownership_status']
# train_x['legal_ownership_status'] = train_x.legal_ownership_status.astype('category')

for col in categorical_columns:
    train_x[col] = pd.Categorical((train_x[col])).codes
    test_x[col] = pd.Categorical((test_x[col])).codes

print('Time started: ' + datetime.now().strftime("%H:%M:%S"))
# X_train, X_test, y_train, y_test = train_test_split(train_x.values[:, 1:], train_y.values[:, 1], test_size=0.33, random_state=42)

# # {'colsample_bytree': 0.7, 'lambda': 1, 'learning_rate': 0.3, 'max_depth': 7, 'min_child_weight': 1, 'missing': -999, 'n_estimators': 250, 'nthread': 4, 'objective': 'binary:logistic', 'seed': 1337, 'subsample': 0.8}
# scaler = StandardScaler()
# train_x_scaled = scaler.fit_transform(train_x.values[:, 1:])
# test_x = scaler.transform(test_x.values)
xgb_model = xgb.XGBClassifier().fit(train_x.values[:, 1:], train_y.values[:, 1])

# print([x for x in np.argsort(xgb_model.feature_importances_) if xgb_model.feature_importances_[x] > 0.008])

# for feat, importance in zip(train_x.columns[1:], xgb_model.feature_importances_):
#     print('feature: {f}, importance: {i}'.format(f=feat, i=importance))

#
# preds = xgb_model.predict(test_x.values[:, 1:])
#
# results = np.zeros((preds.shape[0], 2))
#
# for idx in range(preds.shape[0]):
#     results[idx, :] = [test_x.values[idx, 0], preds[idx]]
#
#
# pd.DataFrame(results).to_csv("DATA/results.csv")


# drop_list = list((36, 35, 32, 34, 31, 33, 29, 5, 12, 7, 6, 30))
# column_name = [train_x.columns[idx+1] for idx in drop_list] + ['building_id']

# test_x = test_x.drop(column_name, axis=1)
#
importances = np.argsort(xgb_model.feature_importances_)
for x in importances:
    print(train_x.columns[1:][x], xgb_model.feature_importances_[x])


# this part was used for feature extraction
# dropped = list(('has_secondary_use_health_post', 'has_secondary_use_use_police', 'has_secondary_use_gov_office', 'has_secondary_use_institution', 'has_secondary_use_industry', 'building_id'))
#
# train_x = train_x.drop(uyyy, axis=1)
# test_x = test_x.drop(uyyy, axis=1)


    # if select_X_train.shape[1] == 26:
    #     preds = selection_model.predict(test_x.values)
    #     results = np.zeros((preds.shape[0], 2))
    #     for idx in range(preds.shape[0]):
    #         results[idx, :] = [test_x.values[idx, 0], preds[idx]]
    #     pd.DataFrame(results).to_csv("DATA/results.csv")

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x.values)
test_x = scaler.transform(test_x.values)
xgb_model = xgb.XGBClassifier().fit(train_x, train_y.values[:, 1])
preds = xgb_model.predict(test_x.values[:, 1:])
results = np.zeros((preds.shape[0], 2))
for idx in range(preds.shape[0]):
    results[idx, :] = [test_x.values[idx, 0], preds[idx]]


pd.DataFrame(preds).to_csv("DATA/results.csv")

print('Time end: ' + datetime.now().strftime("%H:%M:%S"))

# results = []

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for train_index, test_index in kf.split(train_x.values[:, 1:]):
#     xgb_model = xgb.XGBClassifier().fit(train_x.values[train_index, 1:], train_y.values[train_index, 1])
#     scores = xgb_model.score(train_x.values[test_index, 1:], train_y.values[test_index, 1])
#     results.append(scores)
#
# for train_index, test_index in kf.split(train_x.values[:, 1:]):
#     xgb_model = xgb.XGBClassifier().fit(train_x.values[train_index, 1:], train_y.values[train_index, 1])
#     scores = xgb_model.score(train_x.values[test_index, 1:], train_y.values[test_index, 1])
# #     results.append(scores)
#
# print(results)
# print(np.mean(results))
