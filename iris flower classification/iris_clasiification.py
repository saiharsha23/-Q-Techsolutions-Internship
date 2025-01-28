import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



iris=load_iris()
data=pd.DataFrame(data=iris.data,columns=iris.feature_names)
data['species']=iris.target 

print("\ndataset structure")
print(data.head())
print("\nshape of the datset is :")
print(data.shape)
print("\ncolumns of the datset is :")
print(data.columns)
data.to_csv('iris_dataset.csv', index=False)
print(data.isnull().sum())
print(data.describe())

data['species_name'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
sns.pairplot(data, hue='species_name', diag_kind='kde', palette='Set1')
plt.show()



correlation_matrix = data.drop(['species', 'species_name'], axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
x=data[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]

y=data['species']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("\n shapes :")
print(f"X_train:{X_train.shape},X_test:{X_test.shape},y_train:{y_train.shape },y_test:{y_test.shape}")

decisiontree_model = DecisionTreeClassifier(random_state=42)
decisiontree_model.fit(X_train, y_train)
dt_predictions = decisiontree_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("accuracy of the decision tree model :", dt_accuracy)

print("\nClassification Report:\n", classification_report(y_test, dt_predictions))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)


print("\nClassification Report:\n", classification_report(y_test, rf_predictions))




































