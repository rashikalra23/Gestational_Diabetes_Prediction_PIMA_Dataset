from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import load

model = load_model('models/model.h5')

# Accept user input for prediction
Pregnancies = float(input("Enter the number of Pregnancies: "))
Glucose = float(input("Enter the Glucose level: "))
BloodPressure = float(input("Enter the Blood Pressure: "))
SkinThickness = float(input("Enter the Skin Thickness: "))
Insulin = float(input("Enter the Insulin level: "))
BMI = float(input("Enter the BMI: "))
DiabetesPedigreeFunction = float(input("Enter the Diabetes Pedigree Function: "))
Age = float(input("Enter the Age: "))

# Prepare input for prediction
scaler = load('models/scalar.joblib')
input_values = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
input_values = scaler.transform(input_values)
input_values = input_values.reshape(1, input_values.shape[1], 1)

prediction = model.predict(input_values)
print(prediction)
if prediction >= 0.5:
    print("The patient is predicted to have diabetes.")
else:
    print("The patient is predicted not to have diabetes.")