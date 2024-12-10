# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from joblib import dump

# from arch import get_model

# file_path = r"C:\Users\Kalra\Desktop\diabetes.csv"
# data = pd.read_csv(file_path)

# # Split the data into features and labels
# x = data.iloc[:, 0:8].values
# y = data.iloc[:, 8].values

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# dump(scaler, 'models/scalar.joblib')

# # Reshape the data for CNN (add a channel dimension)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# model = get_model(input_shape=(x_train.shape[1], 1))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=35, batch_size=32, validation_data=(x_test, y_test))
# model.save("models/model.h5")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from joblib import dump
from tensorflow.keras.models import load_model

from arch import get_model

# Load dataset
file_path = r"C:\Users\Kalra\Desktop\diabetes.csv"
data = pd.read_csv(file_path)

# Split the data into features and labels
x = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
dump(scaler, 'models/scalar.joblib')

# Reshape the data for CNN (add a channel dimension)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Get the model
model = get_model(input_shape=(x_train.shape[1], 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=35, batch_size=32, validation_data=(x_test, y_test))

# Save the model
model.save("models/model.h5")

# Evaluate the model
print("\nEvaluating the model on the test data...")
y_pred_proba = model.predict(x_test).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")
print(f"ROC-AUC: {roc_auc:.4f}")
