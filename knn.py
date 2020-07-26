from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data=pd.read_csv('DATA/train_values.csv')
data.head()
label=pd.read_csv('DATA/train_labels.csv')
label.head()

data['damage']=label['damage_grade']
data['damage'].value_counts()
X=pd.get_dummies(data.loc[:,:'has_secondary_use_other'])
y=data['damage'].astype(int)

X_test=pd.read_csv('DATA/test_values.csv')
X_test.head()

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X, y)
X_testt = pd.get_dummies(X_test.loc[:, :'has_secondary_use_other'])
y_pred = knn.predict(X_testt)

pd.DataFrame(y_pred).to_csv('knn.csv')
