import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.model_selection import GridSearchCV

dataset = load_breast_cancer()

df = pd.DataFrame(data= dataset.data, columns= dataset.feature_names)
df['target'] = dataset.target

X= df.drop(columns='target' , axis= 'columns')
y = df['target']

# Train Test Split
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size= 0.3, random_state=101)

model = SVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test ,pred))


# Hyperparameter Tuning using GridSearchCV

par_grid = {
           'C' : [0.1,1,10,100,1000,1000], 
           'gamma' : [1 , 0.1 , 0.01 , 0.001 , 0.0001] ,
           'kernel' : ['rbf']
           }

grid = GridSearchCV( SVC() , par_grid , refit= True , verbose= 3)

grid.fit(X_train, y_train)

#print(grid.best_params_)
#print(grid.best_estimator_)

grid_pred = grid.predict(X_test)
print(classification_report(y_test, grid_pred))