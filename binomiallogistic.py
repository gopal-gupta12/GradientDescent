from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y= True)

x_train , x_test , y_train , y_test = train_test_split(X, y , test_size= 0.3, random_state= 23)

mod = LogisticRegression(max_iter= 10000, random_state=0)
mod.fit(x_train, y_train)

pre = mod.predict(x_test)
print(round(accuracy_score(y_test, pre), 2)*100)

