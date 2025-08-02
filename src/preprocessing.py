# src/preprocessing.py

import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Apply preprocessing steps based on EDA findings:
    - Clean categorical columns
    - Handle missing values
    - Clip out-of-range numerical values
    - Add missing indicators

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    df = df.copy()

    # 1. Clean HVAC Operation Mode
    df['HVAC_clean'] = df['HVAC Operation Mode'].str.lower().str.replace('_', ' ').str.strip()
    valid_hvac_modes = ['cooling active', 'eco mode', 'heating active', 'maintenance mode', 'off', 'ventilation only']
    df['HVAC_clean'] = df['HVAC_clean'].astype('category')

    # 2. Clean CO_GasSensor
    df['CO_GasSensor_clean'] = df['CO_GasSensor'].str.lower().str.strip()
    co_order = ['extremely low', 'low', 'medium', 'high', 'extremely high']
    
    # Convert to categorical explicitly
    df['CO_GasSensor_clean'] = pd.Categorical(
        df['CO_GasSensor_clean'],
        categories=co_order,
        ordered=True
    )
    
    # Add 'unknown' category and fill NA
    df['CO_GasSensor_clean'] = df['CO_GasSensor_clean'].cat.add_categories(['unknown'])
    df['CO_GasSensor_clean'] = df['CO_GasSensor_clean'].fillna('unknown')

    # 3. Clean Activity Level
    activity_mapping = {
        'LowActivity': 'Low Activity',
        'Low_Activity': 'Low Activity',
        'ModerateActivity': 'Moderate Activity'
    }
    df['Activity Level'] = df['Activity Level'].replace(activity_mapping)
    # Ensure correct categories
    df['Activity Level'] = pd.Categorical(
        df['Activity Level'],
        categories=['Low Activity', 'Moderate Activity', 'High Activity'],
        ordered=True
    )

    # 4. Clean Ambient Light Level
    df['Ambient Light Level'] = df['Ambient Light Level'].str.lower().str.replace('_', ' ').str.strip()
    light_categories = ['very dim', 'dim', 'moderate', 'bright', 'very bright']
    df['Ambient Light Level'] = pd.Categorical(
        df['Ambient Light Level'],
        categories=light_categories,
        ordered=True
    )
    df['Ambient Light Level'] = df['Ambient Light Level'].cat.add_categories(['unknown'])
    df['Ambient Light Level'] = df['Ambient Light Level'].fillna('unknown')

    # 5. Handle MetalOxideSensor_Unit3: numerical with MNAR missingness
    df['MetalOxideSensor_Unit3_was_missing'] = df['MetalOxideSensor_Unit3'].isnull().astype(int)
    df['MetalOxideSensor_Unit3'] = df['MetalOxideSensor_Unit3'].fillna(df['MetalOxideSensor_Unit3'].median())

    # 6. Clip numerical features to valid ranges
    df['Temperature'] = df['Temperature'].clip(lower=12, upper=40)
    df['Humidity'] = df['Humidity'].clip(lower=20, upper=80)
    df['CO2_ElectroChemicalSensor'] = df['CO2_ElectroChemicalSensor'].clip(lower=400, upper=2000)
    df['CO2_InfraredSensor'] = df['CO2_InfraredSensor'].clip(lower=0, upper=250)

    # 7. Impute CO2_ElectroChemicalSensor if any missing (after clipping)
    df['CO2_ElectroChemicalSensor'] = df['CO2_ElectroChemicalSensor'].fillna(df['CO2_ElectroChemicalSensor'].median())

    print("Preprocessing completed.")
    return df

# Test the function
if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    df_processed = preprocess_data(df)
    print(df_processed.isnull().sum())
    print(df_processed.dtypes)