# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
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
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Step 9: Initialize Model Evaluation Metrics
model_results = {
    "Model": [],
    "Accuracy (%)": [],
    "Precision (%)": [],
    "Recall (%)": [],
    "F1-Score (%)": [],
    "ROC-AUC": [],
    "True Positives": [],
    "True Negatives": [],
    "False Positives": [],
    "False Negatives": []
}

# Step 10: Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42)  # Avoid SAMME.R warning
}

# Step 11: Train and Evaluate Models
for model_name, model in models.items():
    print(f"\nTraining and evaluating model: {model_name}")
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X_resampled, y_resampled), start=1):
        # Split data into train and test for each fold
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
        if roc_auc is not None:
            roc_auc_scores.append(roc_auc)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
        
        # Print metrics for each fold
        print(f"Fold {fold}: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1-Score={f1:.2f}, ROC-AUC={roc_auc:.2f}")
    
    # Store average scores for each model
    model_results["Model"].append(model_name)
    model_results["Accuracy (%)"].append(np.mean(accuracy_scores) * 100)
    model_results["Precision (%)"].append(np.mean(precision_scores) * 100)
    model_results["Recall (%)"].append(np.mean(recall_scores) * 100)
    model_results["F1-Score (%)"].append(np.mean(f1_scores) * 100)
    model_results["ROC-AUC"].append(np.mean(roc_auc_scores) if roc_auc_scores else "N/A")
    model_results["True Positives"].append(np.mean(tp_list))
    model_results["True Negatives"].append(np.mean(tn_list))
    model_results["False Positives"].append(np.mean(fp_list))
    model_results["False Negatives"].append(np.mean(fn_list))

# Step 12: Display Results in a Comparison Table
results_df = pd.DataFrame(model_results)
print("\nComparison Table:")
print(results_df)
