import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objs as go
import datetime

start = '2017-01-01'
end = '2023-11-1'

st.title("Stock Prediction")

user_input = st.text_input("Enter the Stock Ticker", "AAPL")
df = yf.download(user_input, start=start, end=end)

# describing the data
st.subheader("Data from 2017 to 2023")
st.write(df.describe())

# visualization
st.subheader("Closing Price VS Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# visualization for 100 days
st.subheader("Closing Price VS Time Chart with 100 days")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

# visualization for 365 days
st.subheader("Closing Price VS Time Chart with 100 days & 365 days")
ma100 = df.Close.rolling(100).mean()
ma365 = df.Close.rolling(365).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'b')
plt.plot(ma100, 'r')
plt.plot(ma365, 'g')
st.pyplot(fig)

# Load my model
model = load_model("keras model.h5")

# Splitting Data into Training and Testing using MinMaxScaler
data_training = pd.DataFrame(df['Close'][0: int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test);

# prediction making
test_prediction = model.predict(x_test)
# scaler.scale_

scaling_factor = 1 / scaler.scale_[0]
y_test = y_test * scaling_factor
test_prediction = test_prediction * scaling_factor

st.subheader("prediction vs original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(test_prediction, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.title("Next 5 Years Returns")

if user_input:
    symbol = user_input

    try:
        # Download historical data
        stock_data = yf.download(symbol, start="2023-01-01", end="2028-01-01")

        # Calculate returns over the next 5 years
        returns_next_5_years = (stock_data['Adj Close'].iloc[-1] / stock_data['Adj Close'].iloc[0] - 1) * 100

        # Display the returns
        st.write(f"{symbol} returns over the next 5 years: {returns_next_5_years:.2f}%")
        st.write("Recommendation Model")
        if returns_next_5_years >= 0:
            st.markdown('<div style="padding: 10px; color: white; background-color: green; text-align: center;">Yes</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding: 10px; color: white; background-color: red; text-align: center;">No</div>', unsafe_allow_html=True)

    except Exception as e:
        st.write(f"Error: {str(e)}")




