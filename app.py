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
USE_DUMMY_DATA = False

# ThingSpeak Configuration (Required if USE_DUMMY_DATA is False)
# !!! REPLACE WITH YOUR ACTUAL CHANNEL ID AND READ API KEY !!!
TS_CHANNEL_ID = "3158686"
TS_READ_API_KEY = "UYBAXWW4HMKQIO5F"
TS_BASE_URL = f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}/feeds.json?api_key={TS_READ_API_KEY}"
THINGSPEAK_URL = f"{TS_BASE_URL}&results=2"

# TELEGRAM CONFIGURATION (Required for alerts)
# !!! REPLACE WITH YOUR ACTUAL BOT TOKEN AND CHAT ID !!!
TELEGRAM_TOKEN = "8413836677:AAFL1oy4FLv4AQaKuZGxF-nxWfuS_RzkJRQ" 
TELEGRAM_CHAT_ID = "5384764756" 

# =============================================================================
# FLASK SETUP
# =============================================================================

app = Flask(__name__, static_folder='.', static_url_path='/')
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
# TELEGRAM ALERT FUNCTION (New Functionality)
# =============================================================================

def send_telegram_alert(message: str):
    """
    Sends a message to the configured Telegram chat/channel.
    """
    # Safety check for placeholder keys
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
        print("Telegram configuration missing. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    # URL parameters required by Telegram
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        # Send the alert (fire-and-forget request)
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status() 
        print(f"Telegram alert sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram alert: {e}")


# =============================================================================
# DATA LOGIC
# =============================================================================

def calculate_ai_risk(water_level):
    """
    Calculates the AI risk status based on the water level.
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
        return "Low", "LSTM model not available. Using instantaneous risk calculation."

    # 1. Prepare the input data (features: water_level, rainfall)
    water_levels = history_data['water_level']
    rainfalls = history_data['rainfall']
    
    if len(water_levels) < SEQUENCE_LENGTH:
        return "Low", "Insufficient history data for LSTM prediction."

    raw_sequence = np.array(list(zip(water_levels, rainfalls)))[-SEQUENCE_LENGTH:]

    # 2. Scale the data
    try:
        scaled_sequence = scaler.transform(raw_sequence)
    except ValueError as e:
        print(f"Scaling error: {e}. Check if scaler was fitted correctly.")
        return "Low", "Scaling error during LSTM prediction."
    
    # 3. Reshape for LSTM input: (1, sequence_length, n_features)
    X_input = scaled_sequence.reshape(1, SEQUENCE_LENGTH, 2)

    # 4. Predict the next scaled water level
    try:
        scaled_prediction = lstm_model.predict(X_input, verbose=0)[0][0]
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return "Low", "Error running LSTM prediction."

    # 5. Inverse transform the prediction to get the actual water level
    dummy_input = np.zeros((1, 2))
    dummy_input[0, 0] = scaled_prediction
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
        response.raise_for_status() 
        
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
    
    # 0. Initialize alert status for the frontend
    telegram_alert_message = ""
    current_data['telegram_alert_sent'] = ""
    
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
        
    # If instantaneous risk is high, override the message/risk AND TRIGGER ALERT
    if instant_risk == "High":
        final_risk = instant_risk
        final_message = instant_message
        final_recommendation = "IMMEDIATE ACTION: Critical water level detected. Initiate flood mitigation procedures."
        
        # --- TELEGRAM ALERT TRIGGER & DASHBOARD CONFIRMATION ---
        telegram_alert_message = (
            f"🚨 CRITICAL HIGH RISK ALERT 🚨\n"
            f"Time: {current_time}\n"
            f"Water Level: {water_level:.1f} cm\n"
            f"Rainfall: {current_data['rainfall_mm']} mm\n"
            f"Recommendation: {final_recommendation}"
        )
        
        send_telegram_alert(telegram_alert_message)
        current_data['telegram_alert_sent'] = telegram_alert_message
        
    # Update current data with final determined risk
    current_data['ai_risk'] = final_risk
    current_data['ai_message'] = final_message
    current_data['ai_recommendation'] = final_recommendation
    
    state['current'] = current_data
    
    # 5. Generate alerts for the alerts table (using the instantaneous risk for immediate alerts)
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
            
    
    return jsonify(response_data)

def fetch_historical_data(field_id, results=7):
    """Fetches historical data for a specific ThingSpeak field."""
    global TS_CHANNEL_ID, TS_READ_API_KEY
    
    if TS_CHANNEL_ID == "YOUR_CHANNEL_ID" or TS_READ_API_KEY == "YOUR_READ_API_KEY":
        print("ThingSpeak configuration missing for historical data. Returning dummy data.")
        # Return dummy data structure if config is missing
        if field_id == 1: # Rainfall (mm)
            data_points = [random.uniform(0, 20) for _ in range(results)]
            unit = "mm"
        elif field_id == 2: # Water Level (cm)
            data_points = [random.uniform(30, 80) for _ in range(results)]
            unit = "cm"
        elif field_id == 3: # Temperature (°C)
            data_points = [random.uniform(25, 35) for _ in range(results)]
            unit = "°C"
        else:
            data_points = [0] * results
            unit = ""
            
        labels = [f"T-{i*5}m" for i in range(results, 0, -1)] # Mock labels
        return {"labels": labels, "data": [round(d, 1) for d in data_points], "unit": unit}


    # Fetch all fields for the last 'results' entries
    TS_BASE_URL = f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}/feeds.json?api_key={TS_READ_API_KEY}"
    url = f"{TS_BASE_URL}&results={results}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('feeds'):
            return {"labels": [], "data": []}

        labels = []
        field_data = []
        
        field_key = f'field{field_id}'
        
        # Determine unit based on field_id
        if field_id == 1: unit = "mm"
        elif field_id == 2: unit = "cm"
        elif field_id == 3: unit = "°C"
        else: unit = ""
        
        # ThingSpeak returns feeds in chronological order (oldest first)
        for feed in data['feeds']:
            timestamp = feed.get('created_at')
            field_value = feed.get(field_key)
            
            if timestamp and field_value is not None:
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
                    # Use time for recent data, or date if data spans days
                    labels.append(dt.strftime('%H:%M'))
                    field_data.append(round(float(field_value), 1))
                except ValueError:
                    continue
        
        return {"labels": labels, "data": field_data, "unit": unit}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data from ThingSpeak (Field {field_id}): {e}")
        return {"labels": [], "data": []}

@app.route('/api/rainfall', methods=['GET'])
def get_rainfall_history():
    # Fetch 7 data points for Rainfall (Field 1)
    history = fetch_historical_data(field_id=1, results=7)
    return jsonify(history)

@app.route('/api/waterlevel', methods=['GET'])
def get_waterlevel_history():
    # Fetch 7 data points for Water Level (Field 2)
    history = fetch_historical_data(field_id=2, results=7)
    return jsonify(history)

@app.route('/api/temperature', methods=['GET'])
def get_temperature_history():
    # Fetch 7 data points for Temperature (Field 3)
    history = fetch_historical_data(field_id=3, results=7)
    return jsonify(history)


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