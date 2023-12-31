from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

SEED = 23

#loading data
data = pd.read_csv('soc/battery_soc_dataset.csv')
X = data[['Voltage (V)', 'Current (A)', 'Temperature (°C)']]
y = data['SoC'] 

#splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

#creating model
rf = RandomForestRegressor(n_estimators=1000, random_state=SEED) #model

#training data with model
rf.fit(X_train, y_train) #train

#predict test data using trained model
y_pred = rf.predict(X_test)

#metrics
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:{:.2f}".format(mse))

rmse = math.sqrt(mse)
print("Root Mean Squared Error: {:.2f}".format(rmse))

r2 = r2_score(y_test, y_pred)
print("R-squared: {:.2f}".format(r2))

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: {:.2f}".format(mae))

explained_variance = explained_variance_score(y_test, y_pred)
print("Explained Variance Score: {:.2f}".format(explained_variance))

#plot
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual SoC')
plt.ylabel('Predicted SoC')
plt.title('Random Forest: Actual vs Predicted SoC')
plt.show()