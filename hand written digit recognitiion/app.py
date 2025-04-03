from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


d = load_digits()
print("Shape of images in the data set :", d.images.shape)
print("Shape of data in the dataset:", d.data.shape)

 
x = d.data / 16.0  
y = d.target

 
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=10000)
model.fit(train_x, train_y)


print("The accuracy of the model is:", model.score(test_x, test_y))


def visualization_of_predictions(indexing, predicted_values, actual_values):
    for i, idx in enumerate(indexing):
        plt.matshow(test_x[idx].reshape(8, 8), cmap='gray')  # Reshape data to (8x8)
        plt.title(f"Predicted_digits: {predicted_values[i]}, Actual_digits: {actual_values[i]}")
        plt.show()




dig = input("Enter the indices of test rows (comma-separated): ")
rows = list(map(int, dig.split(',')))


 
predict = model.predict(test_x[rows])


print("User-selected indices:", rows)
print("Predicted labels:", predict)
print("Actual labels:", test_y[rows])


visualization_of_predictions(rows, predict, test_y[rows])


