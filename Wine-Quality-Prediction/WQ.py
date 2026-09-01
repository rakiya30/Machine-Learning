#============================================================
# Code Cell
#============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
%matplotlib inline

#============================================================
# Code Cell
#============================================================
wine = pd.read_csv('data.csv')
wine


#============================================================
# Code Cell
#============================================================
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)

#============================================================
# Code Cell
#============================================================
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)

#============================================================
# Code Cell
#============================================================
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

#============================================================
# Code Cell
#============================================================
wine['quality']

#============================================================
# Code Cell
#============================================================
label_quality = LabelEncoder()

#============================================================
# Code Cell
#============================================================
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()

#============================================================
# Code Cell
#============================================================
X = wine.drop('quality', axis = 1)
y = wine['quality']

#============================================================
# Code Cell
#============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#============================================================
# Code Cell
#============================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#============================================================
# Code Cell
#============================================================
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

#============================================================
# Code Cell
#============================================================
print("Accuaracy score =",accuracy_score(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))


#============================================================
# Code Cell
#============================================================
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

#============================================================
# Code Cell
#============================================================
print("Accuaracy score =",accuracy_score(y_test, pred_svc))
print(classification_report(y_test, pred_svc))

#============================================================
# Code Cell
#============================================================
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc,param, cv=10, verbose=2)

#============================================================
# Code Cell
#============================================================
grid_svc.fit(X_train, y_train)

#============================================================
# Code Cell
#============================================================
pred = grid_svc.predict(X_test)
print("Accuaracy score =", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

#============================================================
# Code Cell
#============================================================
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10, verbose=2)
rfc_eval.mean()

#============================================================
# Code Cell
#============================================================

