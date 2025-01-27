import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('./data/housing.csv')

print(data.info())

print(data.isnull().sum())

print(data.head())

label_encoder = LabelEncoder()
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea', 'furnishingstatus']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

print("\nAfter encoding categorical columns:")
print(data.head())

X = data.drop('price', axis=1)  
y = data['price']              

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)


y = np.log1p(y)


poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print("\ndimensions of the data set after adding the polynomial features:", X_poly.shape)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


ridge_param_grid = {'alpha': [0.1, 1.0, 10, 100]}
ridge_cv = GridSearchCV(Ridge(), ridge_param_grid, scoring='r2', cv=5)
ridge_cv.fit(X_train, y_train)

print("\nBest Ridge Alpha:", ridge_cv.best_params_)
print("Best Ridge R2 Score:", ridge_cv.best_score_)
ridge_model = ridge_cv.best_estimator_
ridge_model.fit(X_train, y_train)


y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_ridge}")
print(f"Mean Absolute Error (MAE): {mae_ridge}")
print(f"R-Squared (R2): {r2_ridge}")


lasso_param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0]}
lasso_cv = GridSearchCV(Lasso(), lasso_param_grid, scoring='r2', cv=5)
lasso_cv.fit(X_train, y_train)

print("\nBest Lasso Alpha:", lasso_cv.best_params_)
print("Best Lasso R2 Score:", lasso_cv.best_score_)


lasso_model = lasso_cv.best_estimator_
lasso_model.fit(X_train, y_train)


y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_lasso}")
print(f"Mean Absolute Error (MAE): {mae_lasso}")
print(f"R-Squared (R2): {r2_lasso}")


predictions = pd.DataFrame({
    'Actual': np.expm1(y_test),       
    'Predicted': np.expm1(y_pred_lasso)  
})

output_path = './outputs/predicted_house_prices.csv'
predictions.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")



plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.7, label='Ridge Predictions')
plt.xlabel("Actual Log Prices")
plt.ylabel("Predicted Log Prices")
plt.title("Actual vs Predicted Prices (Ridge Regression)")
plt.legend()
plt.savefig('./visualizations/ridge_plot.png')  
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, alpha=0.7, label='Lasso Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("Actual Log Prices")
plt.ylabel("Predicted Log Prices")
plt.title("Actual vs Predicted Prices (Lasso Regression)")
plt.legend()
plt.savefig('./visualizations/lasso_plot.png')  
plt.show()

import joblib
joblib.dump(ridge_model, './models/ridge_model.pkl')
joblib.dump(lasso_model, './models/lasso_model.pkl')
print("Models saved to the 'models/' folder.")

