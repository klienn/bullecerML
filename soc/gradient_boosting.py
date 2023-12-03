from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import pandas as pd
import math

SEED = 23

data = pd.read_csv('soc/battery_soc_dataset.csv')
X = data[['Voltage (V)', 'Current (A)', 'Temperature (°C)']]
y = data['SoC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

gbr = GradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, n_estimators=300, max_depth = 1, random_state = SEED)

gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)

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