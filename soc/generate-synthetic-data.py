import numpy as np
import pandas as pd

np.random.seed(0)
n_samples = 1000

voltage = np.random.uniform(3.5, 4.2, n_samples)
current = np.random.uniform(-2, 2, n_samples)
temperature = np.random.uniform(25, 45, n_samples)

soc = 0.7 * voltage + 0.1 * current - 0.05 * temperature + np.random.normal(0, 0.1, n_samples)

data = pd.DataFrame({'Voltage (V)': voltage, 'Current (A)': current, 'Temperature (°C)': temperature, 'SoC': soc})

data.to_csv('battery_soc_dataset.csv', index=False)
