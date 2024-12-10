# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 2: Load the Dataset
file_path = r"C:\Users\Kalra\Desktop\diabetes.csv"  # Update with your actual file path
data = pd.read_csv(file_path)

# Step 3: Handle Missing or Implausible Values
columns_with_potential_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zeros with NaN for imputation
data[columns_with_potential_zeros] = data[columns_with_potential_zeros].replace(0, np.nan)

# Apply KNN imputation
imputer = KNNImputer(n_neighbors=5)
data[columns_with_potential_zeros] = imputer.fit_transform(data[columns_with_potential_zeros])

# Step 4: Feature Scaling
scaler = StandardScaler()
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Scale features
X = scaler.fit_transform(X)

# Step 5: Handle Imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Build the Neural Network
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),  # First hidden layer
    Dense(16, activation='relu'),                             # Second hidden layer
    Dense(1, activation='sigmoid')                            # Output layer for binary classification
])

# Step 8: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Step 10: Evaluate the Model
y_pred_prob = model.predict(X_test).ravel()  # Predict probabilities
y_pred = (y_pred_prob > 0.5).astype(int)     # Convert probabilities to binary predictions

# Step 11: Calculate Metrics
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Print Results
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")
print(f"ROC-AUC: {roc_auc:.4f}")

