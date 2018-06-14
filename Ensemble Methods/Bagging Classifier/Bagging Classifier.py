# coding: utf-8

# Bagging Classifier On Dataset of Wharts.

# In[11]:

#Importing Pandas library
import pandas as pd

# In[12]:

#Importing data from excel file
excel = pd.read_excel('cpt.xlsx')

# In[13]:

#Modifying Data & Creating variables X_test ,Y_test ,X_train, Y_train
excel = excel.iloc[:, [0, 1, 3, 4, 5, 6]]
X_train = excel.iloc[0:61, :len(excel.columns) - 1]
Y_train = excel.iloc[0:61, -1]
X_test = excel.iloc[61:, :len(excel.columns) - 1]
Y_test = excel.iloc[61:, -1]

# In[14]:

#Importing Bagging Classifier & Mean Absolute Error Metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_absolute_error as MAE

# In[15]:

#Making bcf Object having Baggi
bcf = BaggingClassifier()
bcf.fit(X_train, Y_train)
#Predicting on X_test
predict = bcf.predict(X_test)

# In[16]:

#Outputting Mean Absolute Error
print("Mean Absolute Error :", MAE(Y_test, predict))
