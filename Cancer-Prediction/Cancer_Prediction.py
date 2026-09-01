import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
data

data.isnull().sum()

data.shape

A=data['diagnosis'].groupby(data['diagnosis']).count()
sns.barplot(x=A.index,y=A.values,data=data)
plt.title("distribution of diagnosis")
plt.show()

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.head()

del data['id']
data.head()

X = data.loc[:,data.columns[1:]]
y = data['diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train) 
y_pred = dt.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

y_pred = dt.predict(X_test) 
acc = accuracy_score(y_test, y_pred)

print(acc)

from sklearn.model_selection import cross_val_score 
import numpy as np

for depth in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
  dt = DecisionTreeClassifier(max_depth=depth) 
  dt.fit(X_train, y_train) 
  trainAccuracy = accuracy_score(y_train, dt.predict(X_train)) 
  dt = DecisionTreeClassifier(max_depth=depth) 
  valAccuracy = cross_val_score(dt, X_train, y_train, cv=10)
  print("Depth  : ", depth, " Training Accuracy : ", trainAccuracy, " Cross val score : " ,np.mean(valAccuracy))

