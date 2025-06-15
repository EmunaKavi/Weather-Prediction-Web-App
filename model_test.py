import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error

# Load the saved model and scaler
model = load_model('weather_lstm_model.h5')
scaler = joblib.load('scaler.save')

# Load and prepare dataset
df = pd.read_csv('Data\weatherHistory.csv')
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])
df = df.sort_values('Formatted Date')

# Use the same 3 features as in training
features = ['Temperature (C)', 'Humidity', 'Pressure (millibars)']
data = df[features].values
scaled_data = scaler.transform(data)

# Create sequences for testing
def create_sequences(data, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps][0])  # Predict Temperature only
    return np.array(X), np.array(y)

time_steps = 24
X, y = create_sequences(scaled_data, time_steps)

# Split into test set (last 20%)
split_index = int(len(X) * 0.8)
X_test = X[split_index:]
y_test = y[split_index:]

# Predict
y_pred = model.predict(X_test)

# Inverse transform temperature only
# We fill humidity and pressure with zeros to match scaler input shape
y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), 2)))))[:, 0]
y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"Test RMSE: {rmse:.2f} °C")

# Plot prediction vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[:500], label='Actual Temperature')
plt.plot(y_pred_inv[:500], label='Predicted Temperature')
plt.title('LSTM Temperature Prediction (Test Set)')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()
