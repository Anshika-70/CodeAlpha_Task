# Install necessary packages if you haven't already
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 1. Load the Pima Indians Diabetes Dataset
# You can download the dataset from Kaggle or use any other dataset of medical records
# For simplicity, we will assume the dataset is in a CSV file called "diabetes.csv"

df = pd.read_csv('diabetes.csv')  # Replace with your dataset path

# 2. Explore the Dataset
print("Dataset Shape:", df.shape)
print(df.head())  # Show first few rows

# 3. Preprocessing
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# For simplicity, let's assume no missing values (if missing values, handle them appropriately)

# Split the data into features (X) and target (y)
X = df.drop('Outcome', axis=1)  # Assuming 'Outcome' is the column with 1 (diabetic) and 0 (non-diabetic)
y = df['Outcome']

# 4. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train the Model using Logistic Regression (you can also try Random Forest, SVM, etc.)

# Logistic Regression
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# 7. Make Predictions and Evaluate the Model
y_pred = log_reg_model.predict(X_test)

# 8. Evaluate Performance
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# 9. Compare with other classifiers like Random Forest or SVM

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# 10. Conclusion: Choose the best model based on accuracy and other evaluation metrics
