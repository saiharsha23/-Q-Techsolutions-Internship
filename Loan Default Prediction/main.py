import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Loan_default.csv")

# Display basic info
print("Dataset Info:\n", df.info())
print("\nMissing values:\n", df.isnull().sum())

# Handling missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical with mode
    else:
        df[col] = df[col].fillna(df[col].median())  # Fill numerical with median

# Confirm no missing values
print("\nMissing values after handling:\n", df.isnull().sum())

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for reference

# Check column names
print("\nColumn names:\n", df.columns)

# Define features (X) and target (y)
target_col = "Default"  # Ensure the correct column name
if target_col in df.columns:
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    raise KeyError(f"Column '{target_col}' not found in dataset!")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# Save predictions to CSV
predictions_df = pd.DataFrame({
    "LoanID": df.iloc[y_test.index]["LoanID"],  # Include LoanID if it's in the dataset
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Save the DataFrame to a CSV file
predictions_df.to_csv("Loan_Default_Predictions.csv", index=False)

print("\nPredictions saved to Loan_Default_Predictions.csv successfully!")
