import random
import requests
import json
import os
from datetime import datetime
from flask import Flask, jsonify, send_file
from flask_cors import CORS

# New imports for LSTM
import numpy as np
import tensorflow as tf # Added for consistency
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib # To load the scaler

# =============================================================================
# CONFIGURATION
# =============================================================================

# Toggle data source:
# True: Use randomly generated dummy data.
# False: Attempt to fetch live data from ThingSpeak.
USE_DUMMY_DATA = True

# ThingSpeak Configuration (Required if USE_DUMMY_DATA is False)
# !!! REPLACE WITH YOUR ACTUAL CHANNEL ID AND READ API KEY !!!
TS_CHANNEL_ID = "3158686"
TS_READ_API_KEY = "UYBAXWW4HMKQIO5F"
TS_BASE_URL = f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}/feeds.json?api_key={TS_READ_API_KEY}"
THINGSPEAK_URL = f"{TS_BASE_URL}&results=50"

# WeatherAPI Configuration (Required for fetching temperature and soil moisture)
# !!! REPLACE WITH YOUR ACTUAL API KEY AND LOCATION !!!
WEATHERAPI_API_KEY = "3ac64ccdd8dc4c1f840190655251811" # Placeholder for WeatherAPI Key
WEATHERAPI_Q = "23.8314,91.2868" # Example: Agartala (lat,lon)
WEATHERAPI_BASE_URL = "http://api.weatherapi.com/v1"

# TELEGRAM CONFIGURATION (Required for alerts)
# !!! REPLACE WITH YOUR ACTUAL BOT TOKEN AND CHAT ID !!!
TELEGRAM_TOKEN = "8413836677:AAFL1oy4FLv4AQaKuZGxF-nxWfuS_RzkJRQ" 
TELEGRAM_CHAT_ID = "5384764756" 

# =============================================================================
# LSTM CONFIGURATION (SYNCHRONIZED FOR 4 FEATURES)
# =============================================================================
LSTM_MODEL_PATH = 'lstm_model.h5'
SCALER_PATH = 'scaler.pkl'
SEQUENCE_LENGTH = 6 # Must match the sequence length used during training
N_FEATURES = 4 # Water Level, Rainfall, Flow Rate, Temperature (Order matters!)

# Global variables to hold the loaded model and scaler
lstm_model = None
scaler = None

def load_lstm_assets():
    """Loads the pre-trained LSTM model and MinMaxScaler."""
    global lstm_model, scaler
    try:
        # Suppress TensorFlow warnings during loading
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
        # We use joblib for the scaler as it's standard for scikit-learn objects
        scaler = joblib.load(SCALER_PATH)
        print("LSTM model and scaler loaded successfully.")
    except Exception as e:
        print(f"Warning: Error loading LSTM assets: {e}. Prediction will be skipped.")
        lstm_model = None
        scaler = None
        
# =============================================================================
# FLASK SETUP
# =============================================================================

app = Flask(__name__, static_folder='.', static_url_path='/')
CORS(app)

DATA_FILE = 'data.json'

def load_data():
    """Loads the entire application state (current, history, alerts) from data.json."""
    
    DEFAULT_HISTORY = {
        "labels": [],
        "rainfall": [],
        "water_level": [],
        "flow_rate": [], # New feature
        "temperature": []    # New feature
    }
    
    DEFAULT_STATE = {
        "current": {},
        "history": DEFAULT_HISTORY,
        "alerts": []
    }
    
    state = DEFAULT_STATE
    
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading data from {DATA_FILE}: {e}. Initializing with default structure.")
            return DEFAULT_STATE
    else:
        print(f"{DATA_FILE} not found. Initializing with default structure.")
        return DEFAULT_STATE

    # Patch history structure if keys are missing (for backward compatibility)
    if 'history' in state:
        history = state['history']
        
        # Explicitly remove old 'soil_moisture' key if present
        if 'soil_moisture' in history:
            del history['soil_moisture']
            
        for key, default_value in DEFAULT_HISTORY.items():
            if key not in history:
                history[key] = default_value
                
    return state

def save_data(data):
    """Saves the entire application state to data.json."""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving data to {DATA_FILE}: {e}")


# =============================================================================
# TELEGRAM ALERT FUNCTION
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
    if water_level > 30:
        risk = "High"
        message = "Critical water level detected. Immediate action required."
    elif 20 <= water_level <= 30:
        risk = "Moderate"
        message = "Water level is elevated. Monitor closely."
    else:
        risk = "Low"
        message = "Water level is normal. System operating within safe parameters."
    
    return risk, message

def predict_lstm_risk(history_data):
    """
    Uses the LSTM model (4 features) to predict the next water level and calculate risk.
    Features order: [water_level_cm, rainfall_mm, flow_rate_lpm, temperature_c]
    """
    global lstm_model, scaler, SEQUENCE_LENGTH, N_FEATURES
    
    if lstm_model is None or scaler is None:
        return "Low", "LSTM model not available. Using instantaneous risk calculation.", None

    # 1. Prepare the input data (4 features)
    water_levels = history_data['water_level']
    rainfalls = history_data['rainfall']
    flow_rates = history_data['flow_rate']
    temperatures = history_data['temperature']
    
    if len(water_levels) < SEQUENCE_LENGTH:
        return "Low", "Insufficient history data for LSTM prediction.", None

    # Zip the last SEQUENCE_LENGTH points of all 4 features
    raw_sequence = np.array(list(zip(
        water_levels, 
        rainfalls, 
        flow_rates,
        temperatures
    )))[-SEQUENCE_LENGTH:]

    # 2. Scale the data
    try:
        scaled_sequence = scaler.transform(raw_sequence)
    except ValueError as e:
        print(f"Scaling error: {e}. Check if scaler was fitted correctly for {N_FEATURES} features.")
        return "Low", "Scaling error during LSTM prediction.", None
    
    # 3. Reshape for LSTM input: (1, sequence_length, n_features)
    X_input = scaled_sequence.reshape(1, SEQUENCE_LENGTH, N_FEATURES)

    # 4. Predict the next scaled water level
    try:
        scaled_prediction = lstm_model.predict(X_input, verbose=0)[0][0]
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return "Low", "Error running LSTM prediction.", None

    # 5. Inverse transform the prediction to get the actual water level
    # We need a dummy input array of shape (1, N_FEATURES) where only the first feature (Water Level) is the prediction.
    dummy_input = np.zeros((1, N_FEATURES))
    dummy_input[0, 0] = scaled_prediction
    
    # Inverse transform requires the full feature set structure
    predicted_water_level = scaler.inverse_transform(dummy_input)[0, 0]
    
    # 6. Calculate risk based on the predicted water level
    predicted_risk, _ = calculate_ai_risk(predicted_water_level)
    
    lstm_message = (
        f"LSTM Forecast (Next Reading): Water level projected to be {predicted_water_level:.1f} cm. "
        f"Predicted risk: {predicted_risk}."
    )
    
    return predicted_risk, lstm_message, round(predicted_water_level, 1)

def get_dummy_data():
    """
    Generates random dummy sensor data, including soil moisture.
    """
    rainfall = round(random.uniform(0.0, 25.0), 1)
    water_level = round(random.uniform(10.0, 90.0), 1)
    temperature = round(random.uniform(20.0, 35.0), 1)
    flow_rate_lpm = random.uniform(0.0, 100.0)
    flow_rate_mlpm = round(flow_rate_lpm * 1000, 1) # Convert L/min to ml/min
    
    risk, message = calculate_ai_risk(water_level)
    
    return {
        "rainfall_mm": rainfall,
        "water_level_cm": water_level,
        "temperature_c": temperature,
        "flow_rate_mlpm": flow_rate_mlpm, # New feature (ml/min)
        "ai_risk": risk,
        "ai_message": message
    }

def get_weatherapi_data():
    """
    Fetches temperature and soil moisture data from WeatherAPI.
    Assumes the API key and location (WEATHERAPI_Q) are configured.
    """
    # Check if configuration is set
    if WEATHERAPI_API_KEY == "YOUR_WEATHERAPI_API_KEY":
        print("WeatherAPI configuration missing. Skipping WeatherAPI fetch.")
        return None

    # Using the forecast endpoint to potentially access soil data (if available in the user's plan)
    # We request 1 day forecast and extract the current hour's data.
    url = f"{WEATHERAPI_BASE_URL}/forecast.json"
    params = {
        "key": WEATHERAPI_API_KEY,
        "q": WEATHERAPI_Q,
        "days": 1,
        "aqi": "no",
        "alerts": "no"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Get current time hour (0-23)
        current_dt = datetime.now()
        current_hour = current_dt.hour

        # Extract data from the forecast for the current hour
        forecast_day = data['forecast']['forecastday'][0]
        
        # Find the data point closest to the current hour
        # We iterate through the 24 hours of forecast data
        current_hour_data = None
        for hour_data in forecast_day['hour']:
            # The time format is 'YYYY-MM-DD HH:MM'
            hour_time_str = hour_data['time'].split(' ')[1].split(':')[0]
            if int(hour_time_str) == current_hour:
                current_hour_data = hour_data
                break
        
        if current_hour_data:
            # Extract temperature (temp_c)
            temperature = float(current_hour_data.get('temp_c', 0.0))
            
            return {
                "temperature_c": temperature
            }
        else:
            print("WeatherAPI: Could not find current hour data in forecast.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from WeatherAPI: {e}. Skipping WeatherAPI data.")
        return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing WeatherAPI response structure: {e}. Skipping WeatherAPI data.")
        return None

def get_thingspeak_data():
    """
    Fetches live data from ThingSpeak (Rainfall, Water Level, Flow Rate) and WeatherAPI (Temperature).
    Falls back to dummy data on critical failure.
    """
    
    # --- 1. Fetch data from WeatherAPI (Temperature) ---
    weather_data = get_weatherapi_data()
    
    # Initialize with fallback values (0.0)
    temperature = 0.0
    
    if weather_data:
        temperature = weather_data['temperature_c']
    else:
        print("WeatherAPI data unavailable. Using 0.0 for temperature.")

    # --- 2. Fetch data from ThingSpeak (Rainfall, Water Level, Flow Rate) ---
    
    # Check if ThingSpeak configuration is set
    if TS_CHANNEL_ID == "YOUR_CHANNEL_ID" or TS_READ_API_KEY == "YOUR_READ_API_KEY":
        print("ThingSpeak configuration missing. Falling back to dummy data for all fields.")
        return get_dummy_data()

    try:
        response = requests.get(THINGSPEAK_URL, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('feeds'):
            print("ThingSpeak returned no feeds. Falling back to dummy data for all fields.")
            return get_dummy_data()

        # Initialize sensor values
        rainfall = 0.0
        water_level = 0.0
        flow_rate = 0.0
        
        # Track which fields have been successfully retrieved
        fields_retrieved = {
            'field1': False, # Rainfall
            'field2': False, # Water Level
            'field4': False  # Flow Rate
        }
        
        # Iterate through feeds (newest first) to find the latest non-null value for each field
        for feed in data['feeds']:
            try:
                if not fields_retrieved['field1'] and feed.get('field1') is not None:
                    rainfall = float(feed['field1'])
                    fields_retrieved['field1'] = True
                    
                if not fields_retrieved['field2'] and feed.get('field2') is not None:
                    water_level = float(feed['field2'])
                    fields_retrieved['field2'] = True
                    
                if not fields_retrieved['field4'] and feed.get('field4') is not None:
                    flow_rate = float(feed['field4'])
                    fields_retrieved['field4'] = True
                    
                # Stop iterating if all required fields are found
                if all(fields_retrieved.values()):
                    break
                    
            except (ValueError, TypeError) as e:
                print(f"Error converting ThingSpeak field value to float in feed {feed.get('entry_id')}: {e}")
                # Continue to the next feed if conversion fails for this one
                continue
        
        # Check if any critical data is still missing (i.e., still 0.0 and not retrieved)
        if not fields_retrieved['field1'] or not fields_retrieved['field2'] or not fields_retrieved['field4']:
            print("Warning: One or more critical ThingSpeak fields were not found in the latest 10 feeds. Using 0.0 for missing values.")
            
        # NOTE: We ignore field3 (if present) as we prioritize WeatherAPI for temperature.

        # --- 3. Combine and Calculate Risk ---
        risk, message = calculate_ai_risk(water_level)
        
        return {
            "rainfall_mm": rainfall,
            "water_level_cm": water_level,
            "temperature_c": temperature,
            "flow_rate_mlpm": round(flow_rate * 1000, 1), # Convert L/min (from ThingSpeak) to ml/min
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
    
    # Ensure history lists are initialized (should be handled by load_data, but safety check)
    if not history.get('labels'):
        history['labels'] = []
        history['rainfall'] = []
        history['water_level'] = []
        history['flow_rate'] = []
        history['temperature'] = []

    # Keep history size manageable (e.g., last 6 entries)
    # Only pop if the history is full AND the list is not empty (for safety/backward compatibility)
    if len(history['labels']) >= MAX_HISTORY_SIZE:
        if history['labels']: history['labels'].pop(0)
        if history['rainfall']: history['rainfall'].pop(0)
        if history['water_level']: history['water_level'].pop(0)
        if history['flow_rate']: history['flow_rate'].pop(0)
        if history['temperature']: history['temperature'].pop(0)
        
    history['labels'].append(current_time)
    history['rainfall'].append(current_data['rainfall_mm'])
    history['water_level'].append(current_data['water_level_cm'])
    history['flow_rate'].append(current_data['flow_rate_mlpm']) # New feature (ml/min)
    history['temperature'].append(current_data['temperature_c']) # New feature
    
    # 2. Calculate instantaneous risk
    instant_risk, instant_message = calculate_ai_risk(water_level)
    
    # 3. Run LSTM prediction using the updated history
    lstm_risk, lstm_message, predicted_level = predict_lstm_risk(history)
    
    # 4. Determine final risk and recommendation based on both instantaneous and predicted risk
    final_risk = instant_risk
    final_message = instant_message
    final_recommendation = "Monitor system closely."
    
    if lstm_model is not None:
        # If LSTM model is available, incorporate its prediction into the final message/recommendation
        
        if lstm_risk == "High":
            # If LSTM predicts high risk, prioritize the predictive alert
            final_recommendation = f"PREDICTIVE ALERT: {lstm_message} Prepare for potential high water levels."
            if instant_risk == "Low":
                # If current risk is low but prediction is high, elevate the overall risk status
                final_risk = "Moderate"
                final_message = f"Water level is currently normal, but predictive model forecasts high risk. ({lstm_message})"
            else:
                # If current risk is already moderate/high, reinforce with prediction
                final_message = f"{instant_message} Predictive model confirms rising risk. ({lstm_message})"
        elif lstm_risk == "Moderate":
            final_recommendation = f"Forecast suggests rising levels. {lstm_message}"
            if instant_risk == "Low":
                final_message = f"Water level is normal, but monitor closely due to forecast. ({lstm_message})"
        else:
            # LSTM predicts Low risk, use instantaneous message and a standard recommendation
            final_recommendation = "System stable. Monitor closely."
            final_message = f"{instant_message} Predictive model confirms low risk."
    else:
        # LSTM model not loaded, use instantaneous message and a standard recommendation
        final_recommendation = "System stable. Monitor closely."
        final_message = instant_message
        
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
            f"Flow Rate: {current_data['flow_rate_mlpm']} ml/min\n" # Include new feature
            f"Temperature: {current_data['temperature_c']} °C\n" # Include new feature
            f"Recommendation: {final_recommendation}"
        )
        
        send_telegram_alert(telegram_alert_message)
        current_data['telegram_alert_sent'] = telegram_alert_message
        
    # Update current data with final determined risk
    current_data['ai_risk'] = final_risk
    current_data['ai_message'] = final_message
    current_data['ai_recommendation'] = final_recommendation
    current_data['ai_predicted_level_cm'] = predicted_level # Store the predicted level explicitly
    
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
        elif field_id == 4: # Flow Rate (ml/min)
            # Generate L/min, then convert to ml/min for consistency with the new API
            data_points_lpm = [random.uniform(0, 100) for _ in range(results)]
            data_points = [round(d * 1000, 1) for d in data_points_lpm]
            unit = "ml/min"
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
        elif field_id == 4: unit = "ml/min" # Unit change
        else: unit = ""
        
        # ThingSpeak returns feeds in chronological order (oldest first)
        for feed in data['feeds']:
            timestamp = feed.get('created_at')
            field_value = feed.get(field_key)
            
            if timestamp and field_value is not None:
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
                    value = float(field_value)
                    
                    # Convert Flow Rate from L/min (ThingSpeak) to ml/min
                    if field_id == 4:
                        value *= 1000
                        
                    # Use time for recent data, or date if data spans days
                    labels.append(dt.strftime('%H:%M'))
                    field_data.append(round(value, 1))
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

@app.route('/api/flowrate', methods=['GET'])
def get_flowrate_history():
    # Fetch 7 data points for Flow Rate (Field 4)
    history = fetch_historical_data(field_id=4, results=7)
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