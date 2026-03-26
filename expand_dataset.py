import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Load existing data
print("Loading existing dataset...")
df = pd.read_csv('loan_data.csv')
original_size = len(df)
print(f"Original size: {original_size}")

# Define target size
target_size = 100000
new_records_needed = target_size - original_size
print(f"Target size: {target_size}")
print(f"Records to generate: {new_records_needed}")

# Get statistics from existing data
income_mean = df['Income'].mean()
income_std = df['Income'].std()
income_min = df['Income'].min()
income_max = df['Income'].max()

loam_mean = df['LoanAmount'].mean()
loan_std = df['LoanAmount'].std()
loan_min = df['LoanAmount'].min()
loan_max = df['LoanAmount'].max()

age_mean = df['Age'].mean()
age_std = df['Age'].std()
age_min = int(df['Age'].min())
age_max = int(df['Age'].max())

credit_mean = df['CreditScore'].mean()
credit_std = df['CreditScore'].std()
credit_min = int(df['CreditScore'].min())
credit_max = int(df['CreditScore'].max())

employment_mean = df['EmploymentYears'].mean()
employment_std = df['EmploymentYears'].std()
employment_min = df['EmploymentYears'].min()
employment_max = df['EmploymentYears'].max()

# Class distribution
class_dist = df['Loan_Status'].value_counts()
class_ratio_0 = class_dist[0] / len(df)
class_ratio_1 = class_dist[1] / len(df)

print(f"\nStatistics from existing data:")
print(f"  Income: mean={income_mean:.0f}, std={income_std:.0f}, range=[{income_min:.0f}, {income_max:.0f}]")
print(f"  LoanAmount: mean={loam_mean:.0f}, std={loan_std:.0f}, range=[{loan_min:.0f}, {loan_max:.0f}]")
print(f"  Age: mean={age_mean:.1f}, std={age_std:.1f}, range=[{age_min}, {age_max}]")
print(f"  CreditScore: mean={credit_mean:.0f}, std={credit_std:.0f}, range=[{credit_min}, {credit_max}]")
print(f"  EmploymentYears: mean={employment_mean:.2f}, std={employment_std:.2f}, range=[{employment_min:.1f}, {employment_max:.1f}]")
print(f"  Class distribution: 0={class_ratio_0:.1%}, 1={class_ratio_1:.1%}")

# Generate synthetic data
print(f"\nGenerating {new_records_needed} synthetic records...")

# Generate Loan_Status first to determine which distribution to use
synthetic_status = np.random.choice([0, 1], size=new_records_needed, p=[class_ratio_0, class_ratio_1])

# Generate features with some variation
synthetic_income = np.random.normal(income_mean, income_std, new_records_needed)
synthetic_income = np.clip(synthetic_income, income_min, income_max)

synthetic_loan = np.random.normal(loam_mean, loan_std, new_records_needed)
synthetic_loan = np.clip(synthetic_loan, loan_min, loan_max)

synthetic_age = np.random.normal(age_mean, age_std, new_records_needed)
synthetic_age = np.clip(synthetic_age, age_min, age_max).astype(int)

synthetic_credit = np.random.normal(credit_mean, credit_std, new_records_needed)
synthetic_credit = np.clip(synthetic_credit, credit_min, credit_max).astype(int)

synthetic_employment = np.random.normal(employment_mean, employment_std, new_records_needed)
synthetic_employment = np.clip(synthetic_employment, employment_min, employment_max)

# Add some correlation for more realistic data
# People with default tend to have lower income, credit score, and employment years
default_mask = synthetic_status == 1
if default_mask.sum() > 0:
    # Reduce income for defaults by ~15%
    synthetic_income[default_mask] *= np.random.uniform(0.85, 0.95, default_mask.sum())
    # Reduce credit score for defaults by ~50-80 points
    synthetic_credit[default_mask] = synthetic_credit[default_mask].astype(float) - np.random.uniform(50, 80, default_mask.sum())
    # Reduce employment years for defaults by ~30%
    synthetic_employment[default_mask] *= np.random.uniform(0.70, 0.90, default_mask.sum())

# Clip again to ensure ranges
synthetic_income = np.clip(synthetic_income, income_min, income_max)
synthetic_credit = np.clip(synthetic_credit, credit_min, credit_max).astype(int)
synthetic_employment = np.clip(synthetic_employment, employment_min, employment_max)

# Create dataframe of synthetic records
synthetic_df = pd.DataFrame({
    'Income': synthetic_income.astype(int),
    'LoanAmount': synthetic_loan.astype(int),
    'Age': synthetic_age,
    'CreditScore': synthetic_credit,
    'EmploymentYears': synthetic_employment.round(1),
    'Loan_Status': synthetic_status.astype(int)
})

# Combine with original data
combined_df = pd.concat([df, synthetic_df], ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# Verify
print(f"\nVerification:")
print(f"  Total records: {len(combined_df)}")
print(f"  Class distribution: {combined_df['Loan_Status'].value_counts().to_dict()}")
print(f"  Class ratio: 0={combined_df['Loan_Status'].value_counts()[0]/len(combined_df):.1%}, 1={combined_df['Loan_Status'].value_counts()[1]/len(combined_df):.1%}")

print(f"\nNew data statistics:")
print(combined_df.describe())

# Save expanded dataset
print(f"\nSaving expanded dataset to loan_data.csv...")
combined_df.to_csv('loan_data.csv', index=False)
print("✓ Successfully saved 100,000 records to loan_data.csv")
