# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define paths
ARTIFACTS_DIR = 'artifacts'

# Load artifacts at startup
model = joblib.load(os.path.join(ARTIFACTS_DIR, 'model.pkl'))
expected_features = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_features.pkl'))
label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data as dictionary
        input_data = request.form.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Convert numeric columns
        numeric_columns = [
            'Temperature', 'Humidity', 'CO2_InfraredSensor',
            'CO2_ElectroChemicalSensor', 'MetalOxideSensor_Unit3'
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Reorder columns to match training data
        df = df[expected_features]

        # Make prediction
        pred_encoded = model.predict(df)[0]
        proba = model.predict_proba(df).max()

        # Decode label
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return jsonify({
            'prediction': pred_label,
            'confidence': round(float(proba), 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)