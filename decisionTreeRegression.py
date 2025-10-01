import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.sort(5 * np.random.rand(1,100), axis= 0)
y = np.sin(X).ravel() + np.random.normal(0 , 0.1, size= X.shape[0])

plt.figure(figsize= (8,5))
plt.scatter(X, y , color = 'red', label = 'data')
plt.title('Synthetic Dataset ')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.tight_layout()
plt.grid(True)
plt.show()