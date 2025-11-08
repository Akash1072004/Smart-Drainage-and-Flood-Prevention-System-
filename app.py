import random
import requests
import json
import os
from datetime import datetime
from flask import Flask, jsonify, send_file
from flask_cors import CORS

# New imports for LSTM
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib # To load the scaler

# Configuration for LSTM
LSTM_MODEL_PATH = 'lstm_model.h5'
SCALER_PATH = 'scaler.pkl'
SEQUENCE_LENGTH = 6 # Must match the sequence length used during training

# Global variables to hold the loaded model and scaler
lstm_model = None
scaler = None

def load_lstm_assets():
    """Loads the pre-trained LSTM model and MinMaxScaler."""
    global lstm_model, scaler
    try:
        # Suppress TensorFlow warnings during loading
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        lstm_model = load_model(LSTM_MODEL_PATH)
        # We use joblib for the scaler as it's standard for scikit-learn objects
        scaler = joblib.load(SCALER_PATH) 
        print("LSTM model and scaler loaded successfully.")
    except Exception as e:
        print(f"Warning: Error loading LSTM assets: {e}. Prediction will be skipped.")
        lstm_model = None
        scaler = None
        
# =============================================================================
# CONFIGURATION
# =============================================================================

# Toggle data source:
# True: Use randomly generated dummy data.
# False: Attempt to fetch live data from ThingSpeak.
USE_DUMMY_DATA = True

# ThingSpeak Configuration (Required if USE_DUMMY_DATA is False)
# !!! REPLACE WITH YOUR ACTUAL CHANNEL ID AND READ API KEY !!!
TS_CHANNEL_ID = "YOUR_CHANNEL_ID"
TS_READ_API_KEY = "YOUR_READ_API_KEY"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}/feeds.json?api_key={TS_READ_API_KEY}&results=1"

# Optional: n8n Webhook URL
# Replace with your n8n webhook URL if you want to send data.
N8N_WEBHOOK_URL = "YOUR_N8N_WEBHOOK_URL"

# =============================================================================
# FLASK SETUP
# =============================================================================

app = Flask(__name__, static_folder='.', static_url_path='/')
# Enable CORS for all routes, allowing the frontend (e.g., running on Live Server)
# to fetch data from this backend running on a different port (e.g., 5000).
CORS(app)

DATA_FILE = 'data.json'

def load_data():
    """Loads the entire application state (current, history, alerts) from data.json."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading data from {DATA_FILE}: {e}. Initializing with empty structure.")
            # Fallback to an empty structure if file is corrupted or unreadable
            return {
                "current": {},
                "history": {"labels": [], "rainfall": [], "water_level": []},
                "alerts": []
            }
    else:
        print(f"{DATA_FILE} not found. Initializing with empty structure.")
        return {
            "current": {},
            "history": {"labels": [], "rainfall": [], "water_level": []},
            "alerts": []
        }

def save_data(data):
    """Saves the entire application state to data.json."""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving data to {DATA_FILE}: {e}")


# =============================================================================
# DATA LOGIC
# =============================================================================

def calculate_ai_risk(water_level):
    """
    Calculates the AI risk status based on the water level.
    
    Logic:
    If water level > 70 -> High risk
    If between 40–70 -> Moderate
    Else -> Low
    """
    if water_level > 70:
        risk = "High"
        message = "Critical water level detected. Immediate action required."
    elif 40 <= water_level <= 70:
        risk = "Moderate"
        message = "Water level is elevated. Monitor closely."
    else:
        risk = "Low"
        message = "Water level is normal. System operating within safe parameters."
    
    return risk, message

def predict_lstm_risk(history_data):
    """
    Uses the LSTM model to predict the next water level and calculate risk.
    """
    global lstm_model, scaler, SEQUENCE_LENGTH
    
    if lstm_model is None or scaler is None:
        # Fallback if model/scaler failed to load (due to missing files or dependency issues)
        return "Low", "LSTM model not available. Using instantaneous risk calculation."

    # 1. Prepare the input data (features: water_level, rainfall)
    # We need the last SEQUENCE_LENGTH points.
    water_levels = history_data['water_level']
    rainfalls = history_data['rainfall']
    
    # Combine into a 2D array: [[WL1, R1], [WL2, R2], ...]
    # Note: We assume the history lists are already padded/truncated to the correct size
    # by update_state_with_current_data, but we ensure we only use the required length.
    
    if len(water_levels) < SEQUENCE_LENGTH:
        return "Low", "Insufficient history data for LSTM prediction."

    # Use only the required sequence length (last SEQUENCE_LENGTH points)
    # We zip and convert to numpy array
    raw_sequence = np.array(list(zip(water_levels, rainfalls)))[-SEQUENCE_LENGTH:]

    # 2. Scale the data
    # NOTE: The scaler must be fitted on the training data for both features (water_level, rainfall).
    try:
        scaled_sequence = scaler.transform(raw_sequence)
    except ValueError as e:
        print(f"Scaling error: {e}. Check if scaler was fitted correctly.")
        return "Low", "Scaling error during LSTM prediction."
    
    # 3. Reshape for LSTM input: (1, sequence_length, n_features)
    X_input = scaled_sequence.reshape(1, SEQUENCE_LENGTH, 2)

    # 4. Predict the next scaled water level
    # We predict the next water level (feature 0)
    try:
        scaled_prediction = lstm_model.predict(X_input, verbose=0)[0][0]
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return "Low", "Error running LSTM prediction."

    # 5. Inverse transform the prediction to get the actual water level
    # We need a dummy array to inverse transform only the water level (first feature)
    # The scaler expects a 2D array with the same number of features it was trained on.
    dummy_input = np.zeros((1, 2))
    dummy_input[0, 0] = scaled_prediction # Put prediction in the water level column
    
    # Inverse transform the water level column
    predicted_water_level = scaler.inverse_transform(dummy_input)[0, 0]
    
    # 6. Calculate risk based on the predicted water level
    predicted_risk, _ = calculate_ai_risk(predicted_water_level)
    
    lstm_message = (
        f"LSTM Prediction: Next water level is projected to be {predicted_water_level:.1f} cm. "
        f"Predicted risk: {predicted_risk}."
    )
    
    return predicted_risk, lstm_message

def get_dummy_data():
    """
    Generates random dummy sensor data.
    """
    rainfall = round(random.uniform(0.0, 25.0), 1)
    water_level = round(random.uniform(10.0, 90.0), 1)
    temperature = round(random.uniform(20.0, 35.0), 1)
    
    risk, message = calculate_ai_risk(water_level)
    
    return {
        "rainfall_mm": rainfall,
        "water_level_cm": water_level,
        "temperature_c": temperature,
        "ai_risk": risk,
        "ai_message": message
    }

def get_thingspeak_data():
    """
    Fetches live data from ThingSpeak. Falls back to dummy data on failure.
    """
    # Check if configuration is set
    if TS_CHANNEL_ID == "YOUR_CHANNEL_ID" or TS_READ_API_KEY == "YOUR_READ_API_KEY":
        print("ThingSpeak configuration missing. Falling back to dummy data.")
        return get_dummy_data()

    try:
        response = requests.get(THINGSPEAK_URL, timeout=5)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        
        if not data.get('feeds'):
            print("ThingSpeak returned no feeds. Falling back to dummy data.")
            return get_dummy_data()

        latest_feed = data['feeds'][0]
        
        # Map fields: field1 -> rainfall, field2 -> water level, field3 -> temperature
        try:
            # Use 0.0 as default if field is missing or None
            rainfall = float(latest_feed.get('field1') or 0.0)
            water_level = float(latest_feed.get('field2') or 0.0)
            temperature = float(latest_feed.get('field3') or 0.0)
        except (ValueError, TypeError):
            print("Error converting ThingSpeak fields to float. Falling back to dummy data.")
            return get_dummy_data()

        risk, message = calculate_ai_risk(water_level)
        
        return {
            "rainfall_mm": rainfall,
            "water_level_cm": water_level,
            "temperature_c": temperature,
            "ai_risk": risk,
            "ai_message": message
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ThingSpeak: {e}. Falling back to dummy data.")
        return get_dummy_data()

def update_state_with_current_data(state, current_data):
    """
    Updates the full application state (history, alerts, messages) based on new current data,
    and incorporates LSTM prediction for advanced risk assessment.
    """
    water_level = current_data['water_level_cm']
    current_time = datetime.now().strftime("%H:%M")
    
    # 1. Update history with the new data point
    history = state['history']
    MAX_HISTORY_SIZE = 6
    
    # Ensure history lists are initialized
    if not history.get('labels'):
        history['labels'] = []
        history['rainfall'] = []
        history['water_level'] = []

    # Keep history size manageable (e.g., last 6 entries)
    if len(history['labels']) >= MAX_HISTORY_SIZE:
        history['labels'].pop(0)
        history['rainfall'].pop(0)
        history['water_level'].pop(0)
        
    history['labels'].append(current_time)
    history['rainfall'].append(current_data['rainfall_mm'])
    history['water_level'].append(current_data['water_level_cm'])
    
    # 2. Calculate instantaneous risk
    instant_risk, instant_message = calculate_ai_risk(water_level)
    
    # 3. Run LSTM prediction using the updated history
    lstm_risk, lstm_message = predict_lstm_risk(history)
    
    # 4. Determine final risk and recommendation based on both instantaneous and predicted risk
    final_risk = instant_risk
    final_message = instant_message
    final_recommendation = "Monitor system closely."
    
    if lstm_model is not None and lstm_risk == "High":
        # If LSTM predicts high risk, prioritize the predictive alert
        final_recommendation = f"PREDICTIVE ALERT: {lstm_message} Prepare for potential high water levels."
        if instant_risk == "Low":
             # If current risk is low but prediction is high, elevate the overall risk status
            final_risk = "Moderate"
            final_message = "Water level is currently normal, but predictive model forecasts high risk."
    elif lstm_model is not None and lstm_risk == "Moderate":
        final_recommendation = f"Forecast suggests rising levels. {lstm_message}"
    else:
        # Use instantaneous message and a standard recommendation
        final_recommendation = "System stable. Monitor closely."
        
    # If instantaneous risk is high, override the message/risk
    if instant_risk == "High":
        final_risk = instant_risk
        final_message = instant_message
        final_recommendation = "IMMEDIATE ACTION: Critical water level detected. Initiate flood mitigation procedures."
        
    # Update current data
    current_data['ai_risk'] = final_risk
    current_data['ai_message'] = final_message
    current_data['ai_recommendation'] = final_recommendation
    
    state['current'] = current_data
    
    # 5. Generate alerts (using the instantaneous risk for immediate alerts)
    alerts = state['alerts']
    
    new_alert = None
    if instant_risk == "High":
        new_alert = { "time": current_time, "zone": "Zone 1", "message": "High flood risk detected.", "risk": "High" }
    elif instant_risk == "Moderate":
        new_alert = { "time": current_time, "zone": "Zone 3", "message": "Water level rising.", "risk": "Moderate" }
        
    if new_alert:
        alerts.insert(0, new_alert)
        # Keep alert list size manageable (e.g., last 10 alerts)
        state['alerts'] = alerts[:10]
        
    return state

def send_to_n8n(data):
    """
    (Optional) Sends the current data payload to an n8n webhook.
    """
    if N8N_WEBHOOK_URL and N8N_WEBHOOK_URL != "YOUR_N8N_WEBHOOK_URL":
        try:
            # Non-blocking send (timeout=1 for quick fire-and-forget)
            requests.post(N8N_WEBHOOK_URL, json=data, timeout=1)
            print("Data successfully sent to n8n webhook.")
        except requests.exceptions.RequestException as e:
            print(f"Error sending data to n8n: {e}")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML file."""
    return send_file('index.html')

@app.route('/data', methods=['GET'])
def get_sensor_data():
    """
    REST API endpoint (/data) to retrieve sensor data based on the configured mode.
    """
    # 1. Get the current state from the file
    state = load_data()
    
    # 2. Get new sensor data
    if USE_DUMMY_DATA:
        new_sensor_data = get_dummy_data()
    else:
        new_sensor_data = get_thingspeak_data()
        
    # 3. Update the state (history, alerts, risk messages)
    updated_state = update_state_with_current_data(state, new_sensor_data)
    
    # 4. Save the updated state back to the file
    save_data(updated_state)
    
    # 5. Prepare response data
    response_data = updated_state
    
    # Ensure current data exists before trying to append message
    if 'current' in response_data:
        if USE_DUMMY_DATA:
            response_data["current"]["ai_message"] += " (Simulated sensor data - DUMMY MODE)."
        else:
            response_data["current"]["ai_message"] += " (Live or fallback sensor data)."
            
        # Optional: Send data to n8n webhook (using the current data part)
        send_to_n8n(response_data["current"])
    
    return jsonify(response_data)

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == '__main__':
    # Load LSTM model assets on startup
    load_lstm_assets()
    
    # How to start the backend server in VS Code terminal:
    # 1. Ensure you have Flask, requests, flask-cors, tensorflow, numpy, scikit-learn, and joblib installed:
    #    pip install Flask requests flask-cors tensorflow numpy scikit-learn joblib
    # 2. Run this file:
    #    python app.py
    # The server will run on http://127.0.0.1:5000/
    app.run(debug=True, port=5000)