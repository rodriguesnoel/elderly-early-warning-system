# train_pipeline.py

import joblib
from pathlib import Path
import pandas as pd
import sys
import json

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from data_loader import load_data
from preprocessing import preprocess_data
from feature_engineer import engineer_features
from model_trainer import train_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    print("Starting training pipeline...")

    # 1. Load data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")

    # 2. Preprocess
    print("Preprocessing data...")
    df = preprocess_data(df)

    # 3. Feature engineer
    print("Engineering features...")
    df = engineer_features(df)

    # 4. Prepare features and target
    print("Preparing X and y...")
    X = df.drop(columns=['Activity Level'])
    y = df['Activity Level']

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Target encoded: {class_mapping}")

    # Ensure all features are float
    X = X.astype(float)

    # Verify no object columns remain
    if (X.dtypes == 'object').any():
        raise ValueError("Object columns remain in X:", X.dtypes[X.dtypes == 'object'])

    # Save feature names for inference
    print("Final feature columns:", X.columns.tolist())

    # 5. Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 6. Train models
    print("Training models...")
    trained_models = train_models(X_train, y_train)
    model = trained_models['LightGBM']
    print("LightGBM selected as best model")
    print("Label encoder classes:", le.classes_)

    # 7. Save artifacts
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)

    print("Saving artifacts to 'artifacts/'...")
    joblib.dump(model, artifacts_dir / 'model.pkl')
    joblib.dump(X_train.columns.tolist(), artifacts_dir / 'model_features.pkl')
    joblib.dump(le, artifacts_dir / 'label_encoder.pkl')

    # --- NEW: Extract and Save Feature Importance ---
    print("Extracting feature importance from LightGBM...")
    importance_values = model.feature_importances_
    feature_names = X_train.columns.tolist()

    # Create sorted importance dictionary
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)

    # Convert to dictionary
    importance_dict = importance_df.set_index('feature')['importance'].to_dict()

    # Save as JSON
    with open(artifacts_dir / 'feature_importance.json', 'w') as f:
        json.dump(importance_dict, f, indent=2)

    print(f"Feature importance saved to {artifacts_dir / 'feature_importance.json'}")

    # Optional: Print top 5 most important features
    print("\nTop 5 Most Important Features:")
    print(importance_df.head(5).to_string(index=False))

    print(f"""
Artifacts saved:
   - Model: {artifacts_dir / 'model.pkl'}
   - Expected feature columns: {artifacts_dir / 'model_features.pkl'}
   - Label encoder: {artifacts_dir / 'label_encoder.pkl'}
   - Feature importance: {artifacts_dir / 'feature_importance.json'}

You're ready to deploy! Run your Flask app next.
""")


if __name__ == "__main__":
    main()