# Import required libraries 
from sklearn.datasets import make_regression 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_features= 1, n_samples= 100 , noise=15 , random_state= 42)
y = y.reshape(-1,1)
m = X.shape[0]
x_feature = np.c_[np.ones((m,1)),X]

theta = np.array([[2.0], [3.0]])

plt.figure(figsize= (8,5))
plt.scatter(X , y , color = 'teal', label = "Actual Data")
plt.plot(X, x_feature.dot(theta) , color = "green", label = "Line (Without Gradient Descent)")
plt.title('Regression Line Without Gradient Descent',fontweight = 'bold')
plt.xlabel('feature', fontweight = 'bold')
plt.ylabel('Target',fontweight = 'bold')
plt.legend()
plt.tight_layout()
plt.show()

learning_rate = 0.1

for _ in range(100):
    pred = x_feature.dot(theta)
    gradient = (2 / m) * x_feature.T.dot(pred - y)
    theta -= learning_rate * gradient


plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, x_feature.dot(theta), color="red", label="Optimized Line (With GD)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression With Gradient Descent")
plt.legend()
plt.show()
