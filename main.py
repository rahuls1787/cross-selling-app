import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Load the dataset
df = pd.read_csv('C:/Users/Administrator/Desktop/dataset_insu.csv')

# Data Preprocessing
# Handle Missing Values
df.ffill(inplace=True)

# Convert Categorical Data using OneHotEncoding
categorical_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
numerical_features = ['Age', 'Annual_Premium', 'Vintage']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply preprocessing to the entire dataset before splitting
X = df.drop('Response', axis=1)
y = df['Response']
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTEENN (combination of SMOTE and Edited Nearest Neighbors)
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

# Define the ensemble model
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
xgb_model = XGBClassifier()
lgbm_model = LGBMClassifier()
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model), ('xgb', xgb_model), ('lgbm', lgbm_model)], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train_res, y_train_res)

# Evaluate the model with threshold adjustment
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find the best threshold
best_threshold = thresholds[(precision + recall).argmax()]

# Apply the best threshold
y_pred = (y_pred_proba >= best_threshold).astype(int)

print(f'Best Threshold: {best_threshold}')
print(classification_report(y_test, y_pred))

import joblib

# Save the preprocessing pipeline
joblib.dump(preprocessor, 'preprocessor.pkl')

# Save the trained ensemble model
joblib.dump(ensemble_model, 'ensemble_model.pkl')

import json

# Convert the trained XGBoost model to JSON
ensemble_model_json = ensemble_model.estimators_[2].get_booster().save_model('ensemble_model.json')