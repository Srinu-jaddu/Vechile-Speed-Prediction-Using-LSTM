#1.import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


#2.looading the data
data = pd.read_excel("C:\\Users\\Srinu Jaddu\\OneDrive\\Desktop\\Traffic_dataset.xlsx")


# Select features/feature engineering
data = data[['Hour', 'Lane 1 Flow (Veh/Hour)', 'Lane 1 Occ (%)', 'Lane 1 Speed (mph)']]


#Visualise the data
# Create subplots with three columns
fig, axs = plt.subplots(1, 3, figsize=(20,8))

# Plot Lane 1 Flow (Veh/Hour)
axs[0].plot(data['Hour'], data['Lane 1 Flow (Veh/Hour)'], label='Lane 1 Flow (Veh/Hour)',c = 'green')
axs[0].set_title('Lane 1 Flow (Veh/Hour) Over Time',fontsize=20)
axs[0].set_xlabel('Hour', fontsize=20) 
axs[0].set_ylabel('Lane 1 Flow (Veh/Hour)', fontsize=20) 
axs[0].legend()

# Plot Lane 1 Occ (%)
axs[1].plot(data['Hour'], data['Lane 1 Occ (%)'], label='Lane 1 Occ (%)',c = 'blue')
axs[1].set_title('Lane 1 Occ (%) Over Time',fontsize=20)
axs[1].set_xlabel('Hour', fontsize=20) 
axs[1].set_ylabel('Lane 1 Occ (%)', fontsize=20)   
axs[1].legend()

# Plot Lane 1 Speed (mph)
axs[2].plot(data['Hour'], data['Lane 1 Speed (mph)'], label='Lane 1 Speed (mph)',c = 'red')
axs[2].set_title('Lane 1 Speed (mph) Over Time',fontsize=20)
axs[2].set_xlabel('Hour', fontsize=20) 
axs[2].set_ylabel('Lane 1 Speed (mph)', fontsize=20)  
axs[2].legend()

plt.tight_layout()
plt.show()


# Normalize the data/ Future Engineering 
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.iloc[:, 1:])  # Exclude 'Hour' column



# Define a function to create sequences
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Build The  LSTM model
def build_lstm_model(seq_length, num_features):
    model = Sequential()
    model.add(LSTM(65, activation='relu', input_shape=(seq_length, num_features)))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


#function to calculate rms Value  to calculate Accuracy
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))



# Create sequences for short-term prediction (1-hour ahead)
seq_length = 24  # Use the previous 24 hours for prediction
X_short_term, y_short_term = create_sequences(data_scaled, seq_length)

# Create sequences for long-term prediction (24-hours ahead)
seq_length = 24  # Use the previous 24 hours for prediction
X_long_term, y_long_term = create_sequences(data_scaled, seq_length)

# Split data into training and testing sets
train_size = int(len(X_short_term) * 0.8)
X_train_short_term, y_train_short_term = X_short_term[:train_size], y_short_term[:train_size]
X_test_short_term, y_test_short_term = X_short_term[train_size:], y_short_term[train_size:]

train_size = int(len(X_long_term) * 0.8)
X_train_long_term, y_train_long_term = X_long_term[:train_size], y_long_term[:train_size]
X_test_long_term, y_test_long_term = X_long_term[train_size:], y_long_term[train_size:]


model_short_term = build_lstm_model(seq_length, 3)  # 3 features excluding 'Hour'
model_long_term = build_lstm_model(seq_length, 3)  # 3 features excluding 'Hour'


print("Short Term Lstm Model is Running :\n")
model_short_term.fit(X_train_short_term, y_train_short_term, epochs=65, batch_size=64)

print("Long Term Lstm Model is Running :\n")
model_long_term.fit(X_train_long_term, y_train_long_term, epochs=65, batch_size=64)

# Evaluate the models
y_pred_short_term = model_short_term.predict(X_test_short_term)
y_pred_long_term = model_long_term.predict(X_test_long_term)

# Inverse transform the scaled data to get the original scale
y_pred_short_term = scaler.inverse_transform(y_pred_short_term)
y_pred_long_term = scaler.inverse_transform(y_pred_long_term)
y_test_short_term = scaler.inverse_transform(y_test_short_term)
y_test_long_term = scaler.inverse_transform(y_test_long_term)

# Calculate and print RMSE
rmse_short_term =calculate_rmse(y_test_short_term, y_pred_short_term) 
rmse_long_term =calculate_rmse(y_test_long_term, y_pred_long_term) 

print(f'Short-Term RMSE: {rmse_short_term}')#to Know the accuracy of short term
print(f'Long-Term RMSE: {rmse_long_term}')#to know the accuracy of long term


# Visualize the results after predictions and before predictions
plt.figure(figsize = (20,10))
plt.plot(y_test_short_term, label='Actual (Short-Term)')
plt.plot(y_pred_short_term, label='Predicted (Short-Term)')
plt.legend(loc = 'upper left',fontsize = 20)
plt.show()

plt.figure(figsize = (20,10))
plt.plot(y_test_long_term, label='Actual (Long-Term)')
plt.plot(y_pred_long_term, label='Predicted (Long-Term)')
plt.legend(loc = 'upper left',fontsize = 20)
plt.show()
