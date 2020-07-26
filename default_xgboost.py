import pandas as pd
from xgboost import XGBClassifier


data=pd.read_csv('DATA/train_values.csv')
data.head()
label=pd.read_csv('DATA/train_labels.csv')
label.head()

data['damage']=label['damage_grade']
data['damage'].value_counts()
X=pd.get_dummies(data.loc[:,:'has_secondary_use_other'])
y=data['damage'].astype(int)

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
