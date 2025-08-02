# src/data_loader.py

import pandas as pd
import sqlite3
import os

def load_data(db_path='data/gas_monitoring.db'):
    """
    Load data from SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please ensure it's in the 'data' folder.")
    
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM gas_monitoring"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Data loaded successfully with shape {df.shape}")
    return df

# Test the function
if __name__ == "__main__":
    df = load_data()
    print(df.head())