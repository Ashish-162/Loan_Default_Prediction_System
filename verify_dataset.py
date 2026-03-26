import pandas as pd

df = pd.read_csv('loan_data.csv')
print('✓ Dataset successfully expanded')
print(f'✓ Total Records: {len(df)}')
print(f'✓ Class Distribution: {df["Loan_Status"].value_counts().to_dict()}')
print(f'✓ Class Ratio: 0={df["Loan_Status"].value_counts()[0]/len(df):.1%}, 1={df["Loan_Status"].value_counts()[1]/len(df):.1%}')
print(f'\nDataset Statistics (summary):')
print(f'  Income: min={df["Income"].min():.0f}, max={df["Income"].max():.0f}, mean={df["Income"].mean():.0f}')
print(f'  LoanAmount: min={df["LoanAmount"].min():.0f}, max={df["LoanAmount"].max():.0f}, mean={df["LoanAmount"].mean():.0f}')
print(f'  Age: min={df["Age"].min()}, max={df["Age"].max()}, mean={df["Age"].mean():.1f}')
print(f'  CreditScore: min={df["CreditScore"].min()}, max={df["CreditScore"].max()}, mean={df["CreditScore"].mean():.0f}')
print(f'  EmploymentYears: min={df["EmploymentYears"].min():.1f}, max={df["EmploymentYears"].max():.1f}, mean={df["EmploymentYears"].mean():.2f}')
