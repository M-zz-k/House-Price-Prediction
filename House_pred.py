# import pandas as pd
# df =pd.read_csv("data.csv")

# df = df.drop(['street','country'],axis = 1)
# df = df.drop(['statezip'], axis=1)
# df['date'] = pd.to_datetime(df['date'])
# df['year'] = df['date'].dt.year
# df['month'] = df['date'].dt.month
# df = df.drop('date',axis = 1)

# df = pd.get_dummies(df,drop_first = True)

# X = df.drop("price",axis = 1)
# y = df['price']

# from sklearn.model_selection import train_test_split
# x_train ,x_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

# # from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
# model.fit(x_train,y_train)
# pred = model.predict(x_test)

# from sklearn.metrics import r2_score,mean_squared_error
# import math
# r2 = r2_score(y_test,pred)
# rms = math.sqrt(mean_squared_error(y_test,pred))
# print("R2: ",r2)
# print("RMS: ",rms)

import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

# Drop useless
df = df.drop(['street','country'], axis=1)

# Handle date
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop('date', axis=1)

# Keep top cities only
top_cities = df['city'].value_counts().nlargest(10).index
df['city'] = df['city'].apply(lambda x: x if x in top_cities else 'Other')

# Remove outliers
df = df[df["price"] < df["price"].quantile(0.99)]

# Log transform
df = df[df["price"] > 0]
df["price"] = np.log(df["price"])

# Encode
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("price", axis=1)
y = df["price"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200)
model.fit(x_train, y_train)

# Predict
pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("R2:", r2)
print("RMSE:", rmse)