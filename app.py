# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)

# Define paths
ARTIFACTS_DIR = 'artifacts'

# Load artifacts at startup
model = joblib.load(os.path.join(ARTIFACTS_DIR, 'model.pkl'))
expected_features = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_features.pkl'))
label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'))

# Load feature importance ---
with open(os.path.join(ARTIFACTS_DIR, 'feature_importance.json'), 'r') as f:
    feature_importance_full = json.load(f)

# Hardcoded class distribution (from training data) ---
# In a real app, you'd save this during training. For now, example values:
class_distribution = {
    'Low': 65,    # % of training data
    'Moderate': 25,
    'High': 10
}

# Optional: You can save this in train_pipeline.py later like:
# with open('artifacts/class_distribution.json', 'w') as f:
#     json.dump(class_distribution_dict, f, indent=2)


@app.route('/')
def home():
    return render_template('index.html')


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

        # --- NEW: Prepare Chart Data ---

        # 1. Top 5 Feature Importance (global, but relevant context)
        top_features = [
            {"feature": k, "importance": v}
            for k, v in sorted(feature_importance_full.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # 2. Class Distribution (static from training)
        class_dist_data = [
            {"class": cls, "percentage": pct}
            for cls, pct in class_distribution.items()
        ]

        # 3. Confidence as percentage
        confidence_percent = round(float(proba) * 100, 1)

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