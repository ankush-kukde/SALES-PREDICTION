import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv('dataset.csv')

# # Scatter Plot
# sns.scatterplot(x='DATE' , y='SALES' , data=data)
# plt.title('Sales over time')
# plt.show()

# # Line Plot
# plt.plot(data.DATE,data.SALES)
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.title('Sales over time')
# plt.show()

"""In this LinePlot, a declining trend of sales over time can be seen. \\
High variability initially and more stable sale values later can be seen.

Please run below cells after preprocessing
"""

# # Applying Rolling Average
# data['Sales_MA7'] = data['SALES'].rolling(window=7).mean()
# plt.plot(data['Date'], data['SALES'], alpha=0.5, label='Original')
# plt.plot(data['Date'], data['Sales_MA7'], label='7-Day Moving Average', color='red')
# plt.legend()
# plt.show()

"""Shows clear Downward trend of Sales over time"""

# # Correlation Matrix
# correlation = final_data.drop(['Sales_MA7','Date_ordinal','Predicted_SALES'],axis=1).corr()
# sns.heatmap(correlation, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

"""From this Heatmap , a correlation of -0.39 can be seen between Date and Sales , meaning with increase in time, sales have decreased but not very strongly.

## Data Preprocessing
"""

# Checking for empty values
# data.isna().sum()

# data['DATE'].dtype

# data['SALES'].dtype

# Converting the Object DATE to Datetime format
data['Date'] = pd.to_datetime(data['DATE'],errors = 'coerce')

# data.head()

data['Date'].dtype

#Just chceking if this method is helpful
data['Date_2'] = pd.to_numeric(data['DATE'], errors='coerce')
# data.head()

# No , it's not!
data.drop('Date_2', axis=1)

# We have our datetime object, we can drop original DATE Coloumn
final_data = data.drop(['DATE','Date_2'], axis=1)

# final_data.head()

final_data['Month'] = final_data['Date'].dt.month
final_data['Year'] = final_data['Date'].dt.year
# sns.boxplot(x='Month', y='SALES', data=final_data)

# sns.boxplot(x='Year', y='SALES', data=final_data)

final_data= final_data.drop(['Month','Year'], axis=1)
print(final_data.head())

# LSTM Method
data_arima = final_data.copy()
data_arima.set_index('Date', inplace=True)
data_lstm = final_data['SALES'].values.reshape(-1, 1)
# data_lstm

scaler = MinMaxScaler()
scaled_data_lstm = scaler.fit_transform(data_lstm)

lookback = 15  # use past 60 days
n_days = int(input('What is the number of days to be forecasted : '))    # forecast next 15 days

X, y = [], []
for i in range(len(scaled_data_lstm) - lookback):
    X.append(scaled_data_lstm[i:i+lookback])
    y.append(scaled_data_lstm[i+lookback])
X, y = np.array(X), np.array(y)
X_train, y_train = X, y

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
    LSTM(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[early_stop])

last_sequence = scaled_data_lstm[-lookback:]  # last 30 days
forecast = []

for _ in range(n_days):
    input_seq = last_sequence.reshape(1, lookback, 1)
    next_val = model.predict(input_seq, verbose=0)[0][0]
    forecast.append(next_val)
    last_sequence = np.append(last_sequence, [[next_val]], axis=0)[-lookback:]

# Inverse transform forecast
forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

data_arima.index = pd.to_datetime(data_arima.index)
future_dates = pd.date_range(start=data_arima.index[-1] + pd.Timedelta(days=1), periods=n_days)

forecast_actual_1D = pd.Series([item[0] for item in forecast_actual])
print(pd.Series(forecast_actual_1D.values, index=future_dates))

# plt.plot(data_arima.index, data_arima['SALES'], label='Original Sales')
# plt.plot(future_dates, forecast_actual, label='Forecast', linestyle='dashed')
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend()
# plt.show()

#This is most accurately predicting model using LSTM

# plt.scatter(data_arima.index, data_arima['SALES'], label='Original Sales')
# plt.scatter(future_dates, forecast_actual, label='Forecast', linestyle='dashed')
# plt.xlabel("Date")
# plt.ylabel("Sales")