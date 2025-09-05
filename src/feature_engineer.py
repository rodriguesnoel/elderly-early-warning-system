# src/feature_engineer.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def engineer_features(df):
    """
    Perform feature engineering: encode categorical variables.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame (output from preprocessing.py)
    
    Returns:
        pd.DataFrame: Model-ready DataFrame with all numerical features
    """
    df = df.copy()
    
    # Initialize OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    # Columns to one-hot encode
    ohe_cols = ['HVAC_clean', 'Ambient Light Level']
    ohe_df = pd.DataFrame(
        ohe.fit_transform(df[ohe_cols]),
        columns=ohe.get_feature_names_out(ohe_cols),
        index=df.index
    )
    
    # Drop original categorical columns and join one-hot encoded ones
    df = df.drop(columns=ohe_cols + ['CO_GasSensor', 'HVAC Operation Mode']).join(ohe_df)
    
    # Ordinal encode CO_GasSensor_clean (already ordered categorical)
    co_mapping = {
        'extremely low': 0,
        'low': 1,
        'medium': 2,
        'high': 3,
        'extremely high': 4,
        'unknown': -1  # preserve missingness
    }
    df['CO_GasSensor_clean_encoded'] = df['CO_GasSensor_clean'].map(co_mapping)
    df = df.drop(columns=['CO_GasSensor_clean'])
    
    # Drop non-informative or redundant columns
    cols_to_drop = ['Time of Day', 'Session ID']  # not predictive of activity, per problem context
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Final check: Ensure no object columns remain
    if (df.dtypes == 'object').any():
        print("⚠️ Warning: Object columns still present:")
        print(df.dtypes[df.dtypes == 'object'])
        raise ValueError("Object columns must be encoded before modeling.")
    
    print("Feature engineering completed.")
    return df

# Test the function
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    print("Final shape:", df.shape)
    print("Feature columns:", df.columns.tolist())