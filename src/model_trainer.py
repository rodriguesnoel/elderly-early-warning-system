# src/model_trainer.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def train_models(X_train, y_train, random_state=42):
    """
    Train multiple classification models.
    
    Args:
        X_train (pd.DataFrame): Engineered feature matrix (numerical)
        y_train (pd.Series): Target labels (Activity Level)
        random_state (int): For reproducibility

    Returns:
        dict: Dictionary of trained models
    """
    models = {}

    # 1. Logistic Regression
    models['LogisticRegression'] = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
        class_weight='balanced',  # Handle class imbalance
        solver='liblinear'       # More stable for datasets with 10k samples and 20+ features
    ).fit(X_train, y_train)

    # 2. Random Forest
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    ).fit(X_train, y_train)

    # 3. LightGBM
    models['LightGBM'] = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        verbosity=-1
    ).fit(X_train, y_train)

    print("All models trained successfully.")
    return models

# Test the function
if __name__ == "__main__":
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.append(str(src_path))
    
    from data_loader import load_data
    from preprocessing import preprocess_data
    from feature_engineer import engineer_features
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Load and process data
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)

    # Define features and target
    X = df.drop(columns=['Activity Level'])
    y = df['Activity Level']

    # Encode target labels as integers
    le = LabelEncoder()
    y = le.fit_transform(y)  # Converts: Low=0, Moderate=1, High=2
    print("Target encoded. Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Debug
    print("\nX_train dtypes:")
    print(X_train.dtypes)
    print("\nAny object columns?")
    print((X_train.dtypes == 'object').sum())

    # Train models
    trained_models = train_models(X_train, y_train)
    print("Trained models:", list(trained_models.keys()))