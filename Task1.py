import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with your actual file path if necessary)
data = pd.read_csv('credit_score_data.csv')  # Make sure the file is in the same folder or give the full path

# Display the first few rows of the data to understand its structure
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values (depending on your strategy)
data.fillna(data.mean(), inplace=True)  # For numerical columns
data['category_column'].fillna(data['category_column'].mode()[0], inplace=True)  # For categorical columns

# Encoding categorical features (if any)
label_encoder = LabelEncoder()
data['categorical_column'] = label_encoder.fit_transform(data['categorical_column'])

# Split data into features (X) and target (y)
X = data.drop(columns=['credit_score'])  # Assuming 'credit_score' is the target column
y = data['credit_score']  # Binary target (e.g., 0 = low, 1 = high creditworthiness)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Random Forest Classifier Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions
logreg_pred = logreg_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test_scaled)

# Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))
print("Logistic Regression Classification Report:\n", classification_report(y_test, logreg_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, logreg_pred))
print("Logistic Regression ROC-AUC Score:", roc_auc_score(y_test, logreg_pred))

# Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Random Forest ROC-AUC Score:", roc_auc_score(y_test, rf_pred))

# Visualize confusion matrix (for Random Forest)
conf_matrix = confusion_matrix(y_test, rf_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Credit', 'High Credit'], yticklabels=['Low Credit', 'High Credit'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest')
plt.show()

# Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("Best Parameters for Random Forest:", grid_search.best_params_)

# Train with best parameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_scaled, y_train)

# Evaluate the optimized Random Forest model
best_rf_pred = best_rf_model.predict(X_test_scaled)
print("Optimized Random Forest Accuracy:", accuracy_score(y_test, best_rf_pred))
print("Optimized Random Forest Classification Report:\n", classification_report(y_test, best_rf_pred))

# Compare accuracies
logreg_accuracy = accuracy_score(y_test, logreg_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Logistic Regression Accuracy: {logreg_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

# Choose the best model based on accuracy (or other metrics)
if logreg_accuracy > rf_accuracy:
    print("Logistic Regression performs better.")
else:
    print("Random Forest performs better.")
