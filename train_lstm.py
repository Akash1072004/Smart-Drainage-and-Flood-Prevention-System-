import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# --- Configuration ---
DATASET_PATH = 'flood_dataset_2000_rows.csv'
MODEL_OUTPUT_PATH = 'lstm_model.h5'
SCALER_OUTPUT_PATH = 'scaler.pkl'
SEQUENCE_LENGTH = 6 # Number of previous time steps to use as input
N_FEATURES = 4 # We are using Water Level, Rainfall, Soil Moisture, and Temperature as features
EPOCHS = 50
BATCH_SIZE = 32

def create_sequences(data, seq_length):
    """Creates sequences of input data and corresponding labels."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Input sequence: last 'seq_length' steps of all N_FEATURES
        X.append(data[i:i + seq_length, :])
        # Output label: the next water level (index 0 in the feature set)
        y.append(data[i + seq_length, 0]) 
    return np.array(X), np.array(y)

def train_model():
    """Loads data, trains the LSTM model, and saves the model and scaler."""
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}. Please ensure the file exists.")
        return

    print(f"Loading data from {DATASET_PATH}...")
    try:
        # Assuming the CSV contains columns like 'water_level_cm', 'rainfall_mm', 'soil_moisture', 'temperature_c'
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- 1. Data Preprocessing ---
    # Select the features used for prediction (Water Level and Rainfall)
    # NOTE: The order matters! Water Level (the target) must be the first column (index 0).
    feature_cols = ['water_level_cm', 'rainfall_mm', 'soil_moisture', 'temperature_c']
    
    # Check if required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}. Please check the dataset structure.")
        return

    data = df[feature_cols].values.astype(float)

    # Initialize and fit the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save the scaler for use in app.py
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"Scaler saved to {SCALER_OUTPUT_PATH}")

    # --- 2. Create Sequences ---
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    
    if len(X) == 0:
        print("Error: Not enough data points to create sequences with the specified SEQUENCE_LENGTH.")
        return

    # --- 3. Build LSTM Model ---
    print("Building LSTM model...")
    model = Sequential([
        # Input shape: (sequence_length, n_features)
        LSTM(50, activation='relu', input_shape=(SEQUENCE_LENGTH, N_FEATURES)),
        Dense(1) # Output layer predicts the next single value (Water Level)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # --- 4. Train Model ---
    print("Training model...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # --- 5. Save Model ---
    tf.keras.models.save_model(model, MODEL_OUTPUT_PATH)
    print(f"Model trained and saved to {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    train_model()