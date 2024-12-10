from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("dataset/diabetes.csv")

# Split the data into features and labels
x = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape the data for CNN (add a channel dimension)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Defining the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=35, batch_size=32)

# Accept user input for prediction
Pregnancies = float(input("Enter the number of Pregnancies: "))
Glucose = float(input("Enter the Glucose level: "))
BloodPressure = float(input("Enter the Blood Pressure: "))
SkinThickness = float(input("Enter the Skin Thickness: "))
Insulin = float(input("Enter the Insulin level: "))
BMI = float(input("Enter the BMI: "))
DiabetesPedigreeFunction = float(input("Enter the Diabetes Pedigree Function: "))
Age = float(input("Enter the Age: "))

input_values = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
input_values = scaler.transform(input_values)
print("Shape of input values before reshape: ", input_values.shape)
input_values = input_values.reshape(1, input_values.shape[1], 1)
print("Shape of input values after reshape: ", input_values.shape)

prediction = model.predict(input_values)
print(prediction)
if prediction >= 0.5:
    print("The patient is predicted to have diabetes.")
else:
    print("The patient is predicted not to have diabetes.")