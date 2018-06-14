
# coding: utf-8

# AdaBoost Regressor On Dataset of Boston House Price

# In[2]:


#Importing Pandas library
import pandas as pd


# In[1]:


#Importing Boston House Data
from sklearn.datasets import load_boston


# In[4]:


#Modifying Data & Creating variables X_test ,Y_test ,X_train, Y_train
boston=load_boston()
bst=pd.DataFrame(boston.data)
X_train=bst.iloc[:400,:]
X_test=bst.iloc[400:,:]
bst=pd.DataFrame(boston.target)
Y_test=bst.iloc[400:,:]
Y_train=bst.iloc[:400,:]


# In[5]:


#Importing AdaBoostRegressor & Mean Absolute Error Metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error as MAE


# In[9]:


#Making abr having AdaBoostRegressor
abr=AdaBoostRegressor()
abr.fit(X_train,Y_train)
predict=abr.predict(X_test)


# In[10]:


#Outputting Mean Absolute Error
print("Mean Absolute Error :",MAE(Y_test,predict))


# In[46]:


#DATA SET IS NOT MODIFIED TO WORK ACCURATELY WITH THE REGRESSOR
#It Just gives an idea about how to Implement the Models

