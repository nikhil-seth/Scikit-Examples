#Predicting House Price Using DT-Regressor
#Import pandas to use csv
import pandas as pd
#Reads CSV
b = pd.read_csv('house-data.csv')
#Reducing No Features in Data as DT-Regressor performs poorly due to overfitting
b = b.iloc[:, [2, 3, 19]]
a = b.iloc[:50, :]
Y = a.iloc[1:, 0]
X = a.iloc[1:, 1:]
#Importing tree from sklearn to use decision Tree Regressor
from sklearn import tree
#rgr object having DT Regressor
rgr = tree.DecisionTreeRegressor()
#Training with X,Y using Fit Function
rgr.fit(X, Y)
#C being test set
C = b.iloc[[28, 60], :]
#rgr Predicting values for C in Test Set
prediction = rgr.predict(C.iloc[:, 1:])
print("Prediction for TestSet\tOrignal Result\n1-", prediction[0], "\t",
      C.iloc[0, 0], "\n2-", prediction[1], "\t", C.iloc[1, 0])

#DT Regressor is prone to overfitting when no of features is high.