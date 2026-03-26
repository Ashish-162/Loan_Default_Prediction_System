#!/usr/bin/env python3
"""
Diagnostic script to identify issues with the loan prediction system
"""

import os
import sys

def print_header(text):
    print(f"\n{'='*60}")
    print(f"{text}")
    print(f"{'='*60}")

def check_files():
    """Check if all required files exist"""
    print_header("1. CHECKING FILES")
    
    files_to_check = {
        'loan_data.csv': 'Dataset',
        'advanced_loan_default_model.pkl': 'Trained Model',
        'app.py': 'Flask App',
        'dashboard.html': 'Dashboard',
        'advanced_loan_default.py': 'Training Script'
    }
    
    all_exist = True
    for filename, description in files_to_check.items():
        exists = os.path.exists(filename)
        status = "✓" if exists else "✗"
        print(f"  {status} {description:20} | {filename}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if required Python packages are installed"""
    print_header("2. CHECKING DEPENDENCIES")
    
    packages = [
        ('flask', 'Flask'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('joblib', 'Joblib'),
        ('imblearn', 'Imbalanced-learn')
    ]
    
    all_installed = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name:20} | Installed")
        except ImportError:
            print(f"  ✗ {name:20} | NOT installed")
            all_installed = False
    
    return all_installed

def check_data_integrity():
    """Check if dataset is valid"""
    print_header("3. CHECKING DATA INTEGRITY")
    
    if not os.path.exists('loan_data.csv'):
        print("  ✗ loan_data.csv not found")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv('loan_data.csv')
        
        required_columns = ['Income', 'LoanAmount', 'Age', 'CreditScore', 'EmploymentYears', 'Loan_Status']
        has_all_columns = all(col in df.columns for col in required_columns)
        
        print(f"  ✓ File loaded successfully")
        print(f"  ✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  {'✓' if has_all_columns else '✗'} Has required columns: {has_all_columns}")
        
        if has_all_columns:
            print(f"     Columns: {list(df.columns)}")
        else:
            missing = [col for col in required_columns if col not in df.columns]
            print(f"     Missing: {missing}")
        
        return has_all_columns
    
    except Exception as e:
        print(f"  ✗ Error reading CSV: {e}")
        return False

def check_model_integrity():
    """Check if model file is valid"""
    print_header("4. CHECKING MODEL INTEGRITY")
    
    if not os.path.exists('advanced_loan_default_model.pkl'):
        print("  ✗ advanced_loan_default_model.pkl not found")
        print("  → Run: python advanced_loan_default.py")
        return False
    
    try:
        import joblib
        model = joblib.load('advanced_loan_default_model.pkl')
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model type: {type(model)}")
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("LOAN DEFAULT PREDICTION SYSTEM - DIAGNOSTIC")
    print("="*60)
    
    # Run checks
    files_ok = check_files()
    deps_ok = check_dependencies()
    data_ok = check_data_integrity()
    model_ok = check_model_integrity()
    
    # Summary
    print_header("SUMMARY")
    
    all_ok = files_ok and deps_ok and data_ok and model_ok
    
    if all_ok:
        print("✓ ALL CHECKS PASSED!")
        print("\n  You can now run:")
        print("  → python app.py")
        print("  → Open http://localhost:5000 in your browser")
        return 0
    
    else:
        print("✗ SOME CHECKS FAILED\n")
        
        if not files_ok:
            print("  FILES ISSUE:")
            print("  → Ensure all files are in the correct directory")
            print()
        
        if not deps_ok:
            print("  DEPENDENCIES ISSUE:")
            print("  → Run: pip install -r requirements.txt")
            print()
        
        if not data_ok:
            print("  DATA ISSUE:")
            print("  → Ensure loan_data.csv has the required columns")
            print("  → Required: Income, LoanAmount, Age, CreditScore, EmploymentYears, Loan_Status")
            print()
        
        if not model_ok:
            print("  MODEL ISSUE:")
            print("  → Run: python advanced_loan_default.py")
            print("  → This will train and save the model")
            print()
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
