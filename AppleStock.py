# -*- coding: utf-8 -*-
import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


client = MongoClient("mongodb://localhost:27017/")
database = client["local"]
column = database["AppleStock"]

data = pd.DataFrame(list(column.find()))
data = data.drop("_id", axis=1)

data["Date"] = pd.to_datetime(data["Date"], format = '%d-%m-%Y')


data['daily_return'] = data['Adj Close'].pct_change()
data.head()
data.isna().sum()



predict_days = 252
data['next_year'] = data['Adj Close'].shift(-predict_days)
data = data.dropna()
x = data[['Adj Close', 'daily_return']]
y = data[['next_year']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)
y_train = y_train.dropna()
x_train.isna().sum()


model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')


data['Day Difference'] = pd.to_timedelta(data['Day Difference'])
latest_date = data['Date'].max()

time_difference = data['Day Difference'].iloc[-1]
next_year_date = latest_date + time_difference

latest_adj_close = data['Adj Close'].iloc[-1]
latest_daily_return = data['daily_return'].iloc[-1]

predicted_price = model.predict([[latest_adj_close, latest_daily_return]])[0]

print(f'Predicted Price on {next_year_date}: {predicted_price}')



