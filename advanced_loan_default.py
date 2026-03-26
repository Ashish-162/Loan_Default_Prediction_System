
# ADVANCED LOAN DEFAULT PREDICTION SYSTEM

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# 1. Load Dataset


data = pd.read_csv("loan_data.csv")

print("Dataset Shape:", data.shape) 
print(data.head())


# 2. Basic Cleaning


data.drop_duplicates(inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)


# 3. Feature Engineering


# Example new feature
if 'Income' in data.columns and 'LoanAmount' in data.columns:
    data['Debt_Income_Ratio'] = data['LoanAmount'] / (data['Income'] + 1)


# 4. Split Features & Target

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

numeric_features = X.select_dtypes(include=['int64','float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


# 5. Preprocessing Pipeline


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 6. Build Model Pipeline with SMOTE


model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=2)),
    ('classifier', model)
])


# 7. Hyperparameter Tuning


param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_


# 8. Model Evaluation


y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print("\nROC-AUC Score:", roc_auc)


# 9. ROC Curve Visualization


fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="XGBoost (AUC = %.3f)" % roc_auc)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
print("ROC Curve saved as 'roc_curve.png'")
plt.close()


# 10. Feature Importance

try:
    importance = best_model.named_steps['classifier'].feature_importances_
    feature_names = list(X.columns)[:len(importance)]
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance[:len(feature_names)]
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop Important Features:")
    print(feature_importance_df)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance.png'")
    plt.close()
    
except Exception as e:
    print(f"Feature importance could not be displayed: {e}")


# 11. Save Model


joblib.dump(best_model, "advanced_loan_default_model.pkl")
print("\nModel Saved Successfully!")
