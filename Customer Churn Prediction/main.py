import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("customer.csv")  


print(df.head())  
print(df.info())  
print(df.isnull().sum()) 


df.drop(columns=['customerID'], inplace=True, errors='ignore')  
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  
df.fillna(0, inplace=True)  

encoder = LabelEncoder()
categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod", "Churn"]

for col in categorical_cols:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])
    else:
        print(f"Warning: Column '{col}' not found in dataset.")


X = df.drop(columns=['Churn'])  
y = df['Churn']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")


log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df, hue="Churn", palette="coolwarm", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()



import joblib
joblib.dump(rf_model, "customer_churn_model.pkl")


results_df = X_test.copy() 
results_df["Actual Churn"] = y_test.values  
results_df["Predicted Churn (Logistic)"] = y_pred_log  
results_df["Predicted Churn (Random Forest)"] = y_pred_rf  


results_df.to_csv("churn_predictions.csv", index=False)
print(" Predictions saved to 'churn_predictions.csv'")

