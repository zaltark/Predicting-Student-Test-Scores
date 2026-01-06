import pandas as pd
import sys
import os
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import load_config

def analyze_dataframe(df, name):
    print(f"\n{'='*30}")
    print(f" Analysis for {name}")
    print(f"{'='*30}")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")
    
    # Column Stats
    stats = {}
    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique())
        }
        
        # Check for empty strings in object columns (potential hidden missing values)
        if df[col].dtype == 'object':
             empty_strings = (df[col].astype(str).str.strip() == "").sum()
             if empty_strings > 0:
                 col_stats["empty_strings"] = int(empty_strings)
                 
        stats[col] = col_stats

    # Print simplified summary table
    print(f"\n{'Column':<20} {'Type':<15} {'Missing':<10} {'Unique':<10} {'% Missing':<10}")
    print("-" * 70)
    for col, data in stats.items():
        pct_missing = (data['missing'] / len(df)) * 100
        print(f"{col:<20} {data['dtype']:<15} {data['missing']:<10} {data['unique']:<10} {pct_missing:.1f}%")
        if "empty_strings" in data:
            print(f"  [WARNING] {col} has {data['empty_strings']} empty/blank strings")

def identify_schema():
    try:
        config = load_config()
        train_path = config['data']['train_path']
        test_path = config['data']['test_path']

        if os.path.exists(train_path):
            print(f"Reading {train_path}...")
            df_train = pd.read_csv(train_path)
            analyze_dataframe(df_train, "Train Data")
        else:
            print(f"File not found: {train_path}")

        if os.path.exists(test_path):
            print(f"Reading {test_path}...")
            df_test = pd.read_csv(test_path)
            analyze_dataframe(df_test, "Test Data")
        else:
            print(f"File not found: {test_path}")

    except Exception as e:
        print(f"Error identifying schema: {e}")

if __name__ == "__main__":
    identify_schema()
