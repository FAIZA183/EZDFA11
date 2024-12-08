# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
# Ensure the dataset file (heart.csv) is in the same directory
data = pd.read_csv("heart.csv")

# Step 2: Data exploration
print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
data.info()
print("\nDataset Description:\n", data.describe())
print("\nMissing Values:\n", data.isnull().sum())

# Step 3: Data visualization
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Visualizing the target variable
sns.countplot(x="target", data=data)
plt.title("Target Variable Distribution")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Count")
plt.show()

# Step 4: Data preprocessing
# Split the dataset into features (X) and target (y)
X = data.drop("target", axis=1)
y = data["target"]

# Normalize the features (optional)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Step 5: Model training
# Using Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 8: Feature importance
# Visualize which features were most important in the prediction
importances = model.feature_importances_
features = data.drop("target", axis=1).columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

# Step 9: Save the model (optional)
# Use pickle to save the model for future use
import pickle
with open("heart_disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel training complete. The model is saved as 'heart_disease_model.pkl'.")
