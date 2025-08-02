# src/model_evaluator.py

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model with classification metrics and visualization.
    
    Args:
        model: Trained classifier
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): True labels
        model_name (str): Name of the model for display
    """
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(f"\n=== {model_name} ===")
    print("Classification Report:")
    class_names = ['High Activity', 'Low Activity', 'Moderate Activity']
    print(classification_report(y_test, y_pred, digits=3))
    
    # Calculate macro F1 (important for imbalanced data)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1-Score: {f1_macro:.3f}")
    
   # Confusion matrix with readable labels
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model_name,
        'macro_f1': f1_macro,
        'y_pred': y_pred
    }

def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return results.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): True labels
    
    Returns:
        list: List of evaluation results
    """
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
    return results

# Test the function
if __name__ == "__main__":
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.append(str(src_path))
    
    from data_loader import load_data
    from preprocessing import preprocess_data
    from feature_engineer import engineer_features
    from model_trainer import train_models
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Load and process data
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)

    # Define features and target
    X = df.drop(columns=['Activity Level'])
    y = df['Activity Level']

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_
    print("Target encoded. Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train models
    trained_models = train_models(X_train, y_train)

    # Evaluate all models
    results = evaluate_all_models(trained_models, X_test, y_test)

    # Optional: Print best model
    best = max(results, key=lambda x: x['macro_f1'])
    print(f"\nBest Model: {best['model']} with Macro F1 = {best['macro_f1']:.3f}")