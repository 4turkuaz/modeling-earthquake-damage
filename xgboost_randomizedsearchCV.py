import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate,KFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

data=pd.read_csv('DATA/train_values.csv')
data.head()
label=pd.read_csv('DATA/train_labels.csv')
label.head()

data['damage']=label['damage_grade']
data['damage'].value_counts()
X=pd.get_dummies(data.loc[:,:'has_secondary_use_other'])
y=data['damage'].astype(int)

'''
This part was used to get best parameters.
Assuming this phase worked previously.

parameters = {
    ’n_jobs’: [-1],
    ’n_estimators’: np.arange(100,1000,100),
    ’max_depth’:np.arange(10, 100, 15),
    ’learning_rate’: [0.03, 0.01, 0.12]
}

clf=XGBClassifier()

kf=KFold(n_splits=2,shuffle=True)

rs=RandomizedSearchCV(clf,param_distributions=param_grid,cv=kf,scoring='f1_micro')

rs.fit(X,y)
print(rs.best_params_)
'''

# using best parameters returned from RandomizedSearchCV
clf=XGBClassifier(n_jobs=-1,n_estimators= 600, max_depth= 10,learning_rate= 0.12)

clf.fit(X,y)

X_test=pd.read_csv('DATA/test_values.csv')
X_test.head()

prediction=clf.predict((pd.get_dummies(X_test)))

result=pd.DataFrame(prediction)

result['building_id']=X_test['building_id']
result.rename(columns={0:'damage_grade'},inplace=True)
result=result[['building_id','damage_grade']]
result.head()
result.to_csv('DATA/result_randomizedsearchCV.csv',index=False)
