from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import pandas as pd
import math

SEED = 32

data = pd.read_csv('soc/battery_soc_dataset.csv')
X = data[['Voltage (V)', 'Current (A)', 'Temperature (°C)']].values
y = data['SoC'].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[1]))

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=SEED)

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:{:.2f}".format(mse))

rmse = math.sqrt(mse)
print("Root Mean Squared Error:{:.2f}".format(rmse))

r2 = r2_score(y_test, y_pred)
print("R-squared:{:.2f}".format(r2))

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:{:.2f}".format(mae))

explained_variance = explained_variance_score(y_test, y_pred)
print("Explained Variance Score:{:.2f}".format( explained_variance))

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('LSTM: Actual vs Predicted SoC')
plt.xlabel('Actual SoC')
plt.ylabel('Predicted SoC')
plt.show()