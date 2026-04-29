# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# ==========================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ==========================================
# Data Cleaning
# ==========================================

# Remove customerID column
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Fill all remaining NaN values
df.fillna(0, inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Encode Target Column
# ==========================================

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ==========================================
# Label Encoding
# ==========================================

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# ==========================================
# Features and Target
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Fill any remaining NaN in X
X = X.fillna(0)

# ==========================================
# Handle Imbalanced Data
# ==========================================

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(pd.Series(y).value_counts())

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# Train Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# Save Model
# ==========================================

pickle.dump(model, open("churn_model.pkl", "wb"))

print("\nModel Saved Successfully")