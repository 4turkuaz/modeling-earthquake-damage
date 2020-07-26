import pandas as pd
import sklearn
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import KFold, GridSearchCV

print('Time started: ' + datetime.now().strftime("%H:%M:%S"))

TRAIN_VALUES = 'DATA/train_values.csv'
TRAIN_LABELS = 'DATA/train_labels.csv'
TEST_VALUES = 'DATA/test_values.csv'

train_x = pd.read_csv(TRAIN_VALUES)
train_y = pd.read_csv(TRAIN_LABELS)
test_x = pd.read_csv(TEST_VALUES)

categorical_columns = list(train_x.columns[8:15]) + ['legal_ownership_status']
for col in categorical_columns:
    train_x[col] = pd.Categorical((train_x[col])).codes
    test_x[col] = pd.Categorical((test_x[col])).codes

xgb_model = xgb.XGBClassifier()

parameters = {'nthread':[4],
              'objective':['binary:logistic'],
              'learning_rate': [0.1],
              'max_depth': [3, 4, 5],
              'min_child_weight': [1],
              'subsample': [0.8],
              'lambda':[0.5, 1],
              'gamma': [1.5, 2, 5],
              'colsample_bytree': [0.8, 1.0],
              'n_estimators': [500, 1000],
              'missing':[-999],
              'seed': [1337]
              }

clf = GridSearchCV(xgb_model, parameters, n_jobs=2,
                   cv=KFold(n_splits=3, shuffle=True, random_state=42),
                   scoring='f1_weighted',
                   verbose=2, refit=True)


clf.fit(train_x.values[:, 1:], train_y.values[:, 1])

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))

print('Time end: ' + datetime.now().strftime("%H:%M:%S"))

# after this phase, use clf.best_params_ and train a model
