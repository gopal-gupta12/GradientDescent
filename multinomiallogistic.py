from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digit = load_digits()

X = digit.data
y = digit.target

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size= 0.25 , random_state= 42)
mod = LogisticRegression(max_iter= 1000, random_state= 0)

mod.fit(x_train , y_train)

pre = mod.predict(x_test)

print(round(accuracy_score(y_test, pre),2))