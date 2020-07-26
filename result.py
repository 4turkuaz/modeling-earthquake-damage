import os
import sklearn
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')


train_x = pd.read_csv('train_values.csv')
train_y = pd.read_csv('train_labels.csv')
test_x = pd.read_csv('test_values.csv')

print(train_x.shape)
print(train_y.shape)




print(train_x[:2])
print(train_y[:2])
# print(train_y.shape)

# print(train_x[0, 8:15])
categorical_columns = list(train_x.columns[8:15]) + ['legal_ownership_status']
# train_x['legal_ownership_status'] = train_x.legal_ownership_status.astype('category')

for col in categorical_columns:
    train_x[col] = pd.Categorical((train_x.__getitem__(col))).codes
    test_x[col] = pd.Categorical((test_x.__getitem__(col))).codes

# pd.Categorical.from_array(dataframe.col3).codes

# print(train_x.values[:2])
# exit()

# X_train, X_test, y_train, y_test = train_test_split(train_x.values[:, 1:], train_y.values[:, 1], test_size=0.2, random_state=42)

# results = []



# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['binary:logistic'],
#               'learning_rate': [0.3], #so called `eta` value
#               'max_depth': [6,20],
#               'min_child_weight': [1,5],
#               'subsample': [0.8],
#               'lambda':[0.5, 1],
#               'colsample_bytree': [0.7],
#               'n_estimators': [250], #number of trees, change it to 1000 for better results
#               'missing':[-999],
#               'seed': [1337]}

# xgb_model = xgb.XGBClassifier()

# clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
#                    cv=KFold(n_splits=5, shuffle=True, random_state=42), 
#                    scoring='accuracy',
#                    verbose=2, refit=True)


# clf.fit(train_x.values[:, 1:], train_y.values[:, 1])


# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#             % (mean, std * 2, params))

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for train_index, test_index in kf.split(train_x.values[:, 1:]):
#     xgb_model = xgb.XGBClassifier().fit(train_x.values[train_index, 1:], train_y.values[train_index, 1])
#     scores = xgb_model.score(train_x.values[test_index, 1:], train_y.values[test_index, 1])
#     results.append(scores)


xgb_model = xgb.XGBClassifier(njobs=4, max_depth = 10).fit(train_x.values[:, 1:], train_y.values[:, 1])
preds = xgb_model.predict(test_x.values[:, 1:])

results = np.zeros((preds.shape[0], 2))

for idx in range(preds.shape[0]):
    results[idx, :] = [test_x.values[idx, 0], preds[idx]]


pd.DataFrame(results).to_csv("results.csv")


# print(results)
# print(np.mean(results))

# eclf1 = eclf1.fit(X_train, y_train)
# scores = eclf1.score(X_test, y_test)

# print(scores)





