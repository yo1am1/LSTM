import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

import matplotlib.pyplot as plt

data = pd.read_csv("prices-split-adjusted.csv")

features = ["open", "close", "low", "high", "volume"]
data = data[features]

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

window_size = 10
sequences = []
target = []
for i in range(len(data) - window_size):
    sequences.append(data_normalized[i : i + window_size])
    target.append(data_normalized[i + window_size])

X = np.array(sequences)
y = np.array(target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(
    Bidirectional(
        LSTM(
            50,
            activation="relu",
            input_shape=(window_size, len(features)),
            dropout=0.2,
            return_sequences=True,
        )
    )
)
model.add(LSTM(50, activation="relu", dropout=0.2))
model.add(Dense(len(features)))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Set: {loss}")

predictions = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test)

plt.figure(figsize=(15, 8))

plt.plot(actual_prices, label="Actual Prices", color="blue")

plt.plot(predicted_prices, label="Predicted Prices", color="orange")

plt.title("Stock Price Prediction with Bidirectional LSTM and Stacked LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
