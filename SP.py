# --------------------------------------------------
import pandas as pd 
import numpy as np
import matplotlib as plt 
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

# --------------------------------------------------
df = pd.read_csv("TSLA.csv")
df

# --------------------------------------------------
df = df.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                'Adj Close':'adj_close','Volume':'volume'})
df.head()

# --------------------------------------------------
df['date'] = pd.to_datetime(df.date)

# --------------------------------------------------

fig= px.line(df, x='date', y='close', title='Tesla Close Price')
fig.show()

# --------------------------------------------------
closedf = df[['date','close']]
print("Shape of close dataframe:", closedf.shape)

# --------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf['close']).reshape(-1,1))
print(closedf.shape)

# --------------------------------------------------
training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

# --------------------------------------------------
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# --------------------------------------------------
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# --------------------------------------------------
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(X_train, y_train, verbose=True)

# --------------------------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# --------------------------------------------------
predictions = my_model.predict(X_test)
print("Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions)))
print("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions))))

# --------------------------------------------------

