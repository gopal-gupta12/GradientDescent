import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

df = pd.DataFrame(data= iris.data, columns= iris.feature_names)

df['target'] = iris.target

df['flower-name'] = df.target.apply(lambda x : iris.target_names[x])


df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

#plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color = 'red', marker= '+')
#plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color = 'green', marker= '.')

#plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color = 'red', marker= '+')
#plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color = 'green', marker= '.')
#plt.tight_layout()
#plt.grid(True)
#plt.show()

X = df.drop(['target', 'flower-name'], axis= 'columns')
y = df.target
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size= 0.2 )
model = SVC(kernel= 'linear')
model.fit(X_train, y_train)
print(model.score(X_test, y_test)*100)