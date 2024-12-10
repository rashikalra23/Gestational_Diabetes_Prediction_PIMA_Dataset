# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 2: Load the Dataset
file_path = r"C:\Users\Kalra\Desktop\diabetes.csv"
data = pd.read_csv(file_path)

# Step 3: Perform Exploratory Data Analysis (EDA)
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

# Step 4: Check for Zeros in Critical Columns
columns_with_potential_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nCount of zeros in critical columns:")
print((data[columns_with_potential_zeros] == 0).sum())

# Step 5: Handle Missing Values (Using KNN Imputer)
data[columns_with_potential_zeros] = data[columns_with_potential_zeros].replace(0, np.nan)
imputer = KNNImputer(n_neighbors=5)
data[columns_with_potential_zeros] = imputer.fit_transform(data[columns_with_potential_zeros])
print("\nData after KNN imputation (first 5 rows):")
print(data.head())

# Step 6: Normalize Continuous Variables
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['Outcome']))
data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])
data_scaled['Outcome'] = data['Outcome']
print("\nData after scaling (first 5 rows):")
print(data_scaled.head())

# Step 7: Split Features and Target
X = data_scaled.drop(columns=['Outcome'])
y = data_scaled['Outcome']

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("\nClass Distribution After SMOTE:")
print(y_resampled.value_counts())

# Step 8: Initialize k-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 9: Train Original Random Forest
original_rf = RandomForestClassifier(random_state=42)
accuracy_scores = []

for train_index, test_index in kf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    
    original_rf.fit(X_train, y_train)
    y_pred = original_rf.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

original_rf_accuracy = np.mean(accuracy_scores) * 100
print(f"\nOriginal Random Forest Accuracy: {original_rf_accuracy:.2f}%")

# Step 10: Hyperparameter Tuning for Random Forest using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    verbose=2,
    n_jobs=-1
)

print("Performing Grid Search for Hyperparameter Tuning...")
grid_search.fit(X_resampled, y_resampled)

# Extract the best parameters
best_params = grid_search.best_params_
print("\nBest Parameters from Grid Search:")
print(best_params)

# Train the Random Forest with best parameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_resampled, y_resampled)

# Evaluate the tuned model using Stratified k-Fold Cross-Validation
accuracy_scores = []

for train_index, test_index in kf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

tuned_rf_accuracy = np.mean(accuracy_scores) * 100
print(f"\nTuned Random Forest Accuracy: {tuned_rf_accuracy:.2f}%")

# Step 11: Compare Results
print("\nComparison of Results:")
print(f"Original Random Forest Accuracy: {original_rf_accuracy:.2f}%")
print(f"Tuned Random Forest Accuracy: {tuned_rf_accuracy:.2f}%")
