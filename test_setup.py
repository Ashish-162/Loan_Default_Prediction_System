#!/usr/bin/env python3
"""
Test script to verify the loan default prediction system setup
"""

import os
import sys

def check_model_exists():
    """Check if the trained model exists"""
    if os.path.exists('advanced_loan_default_model.pkl'):
        print("✓ Model file found: advanced_loan_default_model.pkl")
        return True
    else:
        print("✗ Model file NOT found")
        print("  Run this command first: python advanced_loan_default.py")
        return False

def check_dataset_exists():
    """Check if the dataset exists"""
    if os.path.exists('loan_data.csv'):
        print("✓ Dataset found: loan_data.csv")
        import pandas as pd
        df = pd.read_csv('loan_data.csv')
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        return True
    else:
        print("✗ Dataset NOT found: loan_data.csv")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'flask': 'Flask',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'joblib': 'joblib'
    }
    
    all_installed = True
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            all_installed = False
    
    return all_installed

def main():
    print("=" * 60)
    print("LOAN DEFAULT PREDICTION SYSTEM - SETUP CHECK")
    print("=" * 60)
    
    print("\n1. Checking Dependencies...")
    deps_ok = check_dependencies()
    
    print("\n2. Checking Dataset...")
    dataset_ok = check_dataset_exists()
    
    print("\n3. Checking Model...")
    model_ok = check_model_exists()
    
    print("\n" + "=" * 60)
    
    if deps_ok and dataset_ok and model_ok:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou can now run: python app.py")
        print("Then open: http://localhost:5000")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        if not deps_ok:
            print("\n  Fix: pip install -r requirements.txt")
        if not dataset_ok:
            print("\n  Fix: Ensure loan_data.csv is in the project folder")
        if not model_ok:
            print("\n  Fix: python advanced_loan_default.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())
