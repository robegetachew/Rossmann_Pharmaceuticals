import numpy as np
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_time_series(data):
    data = data.set_index('Date').resample('D').sum().fillna(0)
    return data

def check_stationarity(data):
    result = adfuller(data['Sales'])
    return result[1] < 0.05  # p-value < 0.05 means stationary

def create_supervised_data(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])  # Keep it as 1D
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def train_lstm_model(model, X_train, y_train, epochs=100, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)