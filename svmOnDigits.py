import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digit = load_digits()
df = pd.DataFrame(data= digit.data , columns= digit.feature_names)
df['target'] = digit.target

X = df.drop(columns= 'target', axis='columns')
y = df['target']

X_train  , X_test , y_train , y_test = train_test_split(X, y , test_size= 0.2)
model = SVC(C= 7)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)*100)