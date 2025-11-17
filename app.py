from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
import json
import mysql.connector
from datetime import datetime

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'srv1865.hstgr.io'
app.config['MYSQL_USER'] = 'u253172392_early_warning'
app.config['MYSQL_PASSWORD'] = 't&93UtnA'
app.config['MYSQL_DB'] = 'u253172392_early_warning'
app.config['MYSQL_PORT'] = 3306

def get_db_connection():
    """Create and return MySQL connection"""
    try:
        conn = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT']
        )
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

def init_database():
    """Initialize database tables if they don't exist"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            
            # Create predictions table for storing new predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255),
                    temperature FLOAT,
                    humidity FLOAT,
                    co2_infrared FLOAT,
                    co2_electrochemical FLOAT,
                    metal_oxide_sensor FLOAT,
                    co_gas_sensor VARCHAR(50),
                    hvac_mode VARCHAR(50),
                    ambient_light VARCHAR(50),
                    prediction VARCHAR(50),
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Database initialized successfully")
        except mysql.connector.Error as e:
            print(f"❌ Database initialization error: {e}")

# Initialize database on startup
init_database()

# Define paths
ARTIFACTS_DIR = 'artifacts'

# Load artifacts at startup
print("Loading ML artifacts...")
model = joblib.load(os.path.join(ARTIFACTS_DIR, 'model.pkl'))
expected_features = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_features.pkl'))
label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'))

# Load feature importance
with open(os.path.join(ARTIFACTS_DIR, 'feature_importance.json'), 'r') as f:
    feature_importance_full = json.load(f)

# Hardcoded class distribution (from training data)
class_distribution = {
    'Low': 65,    # % of training data
    'Moderate': 25,
    'High': 10
}

print("✅ All artifacts loaded successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test-db')
def test_db():
    """Test database connection"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return jsonify({'status': '✅ Database connection successful!'})
        else:
            return jsonify({'status': '❌ Database connection failed!'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def log_prediction_to_db(input_data, prediction, confidence):
    """Log prediction to MySQL database"""
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to database for logging")
        return
    
    try:
        cursor = conn.cursor()
        
        # Insert prediction record
        cursor.execute('''
            INSERT INTO predictions 
            (session_id, temperature, humidity, co2_infrared, co2_electrochemical, 
             metal_oxide_sensor, co_gas_sensor, hvac_mode, ambient_light, prediction, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            input_data.get('Session ID', 'unknown'),
            input_data.get('Temperature'),
            input_data.get('Humidity'),
            input_data.get('CO2_InfraredSensor'),
            input_data.get('CO2_ElectroChemicalSensor'),
            input_data.get('MetalOxideSensor_Unit3'),
            input_data.get('CO_GasSensor'),
            input_data.get('HVAC_Operation_Mode'),
            input_data.get('Ambient_Light_Level'),
            prediction,
            confidence
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Prediction logged to database successfully")
    except Exception as e:
        print(f"❌ Error logging prediction to database: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()
        print("Received form data:", input_data)

        # === MAP DESCRIPTIVE STRINGS TO NUMERICAL VALUES ===
        temperature_map = {
            'Below 12°C - Dangerously Cold': 10,
            '12–20°C - Cold': 16,
            '20–26°C - Cool': 23,
            '26–30°C - Normal': 28,
            '30–34°C - Warm': 32,
            '34–40°C - Hot': 37,
            'Above 40°C - Dangerously Hot': 42
        }

        humidity_map = {
            'Below 20% - Dangerously Dry': 15,
            '20–40% - Dry': 30,
            '40–60% - Comfortable': 50,
            '60–80% - Humid': 70,
            'Above 80% - Dangerously Humid': 85
        }

        co2_ir_map = {
            '0–50 - Very Low': 25,
            '50–125 - Moderate': 87,
            '125–200 - High': 162,
            '200–250 - Very High': 225
        }

        co2_ec_map = {
            '400–600 - Normal': 500,
            '600–800 - Slightly Elevated': 700,
            '800–1000 - Elevated': 900,
            '1000–1500 - High': 1250,
            '1500–2000 - Very High': 1750
        }

        mox_map = {
            '0–200 - Low': 100,
            '200–500 - Moderate': 350,
            '500–800 - High': 650,
            '800–1000 - Very High': 900,
            'nan': None
        }

        # Convert descriptive strings to numerical values
        df = pd.DataFrame([input_data])
        df['Temperature'] = df['Temperature'].map(temperature_map)
        df['Humidity'] = df['Humidity'].map(humidity_map)
        df['CO2_InfraredSensor'] = df['CO2_InfraredSensor'].map(co2_ir_map)
        df['CO2_ElectroChemicalSensor'] = df['CO2_ElectroChemicalSensor'].map(co2_ec_map)
        df['MetalOxideSensor_Unit3'] = df['MetalOxideSensor_Unit3'].map(mox_map)

        # Validate required fields
        required_fields = [
            'Temperature', 'Humidity', 'CO2_InfraredSensor',
            'CO2_ElectroChemicalSensor', 'MetalOxideSensor_Unit3',
            'CO_GasSensor', 'HVAC_Operation_Mode', 'Ambient_Light_Level'
        ]
        for field in required_fields:
            if field not in df:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Check if all numerical inputs are missing
        numerical_fields = [
            'Temperature', 'Humidity', 'CO2_InfraredSensor',
            'CO2_ElectroChemicalSensor', 'MetalOxideSensor_Unit3'
        ]
        if df[numerical_fields].isna().all(axis=1).iloc[0]:
            return jsonify({'error': 'All sensor readings are missing. Please provide at least one valid input.'}), 400

        # === REPLAY PREPROCESSING ===
        df['Temperature'] = df['Temperature'].clip(lower=12, upper=40)
        df['Humidity'] = df['Humidity'].clip(lower=20, upper=80)
        df['CO2_InfraredSensor'] = df['CO2_InfraredSensor'].clip(lower=0, upper=250)
        df['CO2_ElectroChemicalSensor'] = df['CO2_ElectroChemicalSensor'].clip(lower=400, upper=2000)

        df['MetalOxideSensor_Unit3_was_missing'] = df['MetalOxideSensor_Unit3'].isna().astype(int)
        df['MetalOxideSensor_Unit3'] = df['MetalOxideSensor_Unit3'].fillna(df['MetalOxideSensor_Unit3'].median())

        # Clean categorical variables
        df['CO_GasSensor'] = df['CO_GasSensor'].str.lower().str.strip()
        df['HVAC_Operation_Mode'] = (
            df['HVAC_Operation_Mode']
            .str.lower()
            .str.replace('_', ' ')
            .str.replace('-', ' ')
            .str.strip()
        )
        df['Ambient_Light_Level'] = df['Ambient_Light_Level'].str.lower().str.strip()

        df['CO_GasSensor'] = df['CO_GasSensor'].replace('', 'unknown').fillna('unknown')
        df['HVAC_Operation_Mode'] = df['HVAC_Operation_Mode'].replace('', 'unknown').fillna('unknown')
        df['Ambient_Light_Level'] = df['Ambient_Light_Level'].replace('', 'unknown').fillna('unknown')

        # === FEATURE ENGINEERING ===
        ohe_cols = ['HVAC_Operation_Mode', 'Ambient_Light_Level']
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

        co_mapping = {
            'extremely low': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'extremely high': 4,
            'unknown': -1
        }
        df['CO_GasSensor_clean_encoded'] = df['CO_GasSensor'].map(co_mapping)
        df = df.drop(columns=['CO_GasSensor'], errors='ignore')

        cols_to_drop = ['Time of Day', 'Session ID']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        # Ensure all expected features are present
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]
        df = df.astype(float)

        # === PREDICTION ===
        pred_encoded = model.predict(df)[0]
        proba = model.predict_proba(df).max()
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        # === LOG PREDICTION TO MYSQL DATABASE ===
        log_prediction_to_db(input_data, pred_label, float(proba))

        # Prepare response data
        confidence_percent = round(float(proba) * 100, 1)
        top_features = [
            {"feature": k, "importance": v}
            for k, v in sorted(feature_importance_full.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        class_dist_data = [
            {"class": cls, "percentage": pct}
            for cls, pct in class_distribution.items()
        ]

        return jsonify({
            'prediction': pred_label,
            'confidence': round(float(proba), 3),
            'confidence_percent': confidence_percent,
            'feature_importance': top_features,
            'class_distribution': class_dist_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)