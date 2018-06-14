
# coding: utf-8

# Extra Trees Regressor on Data Set of House Prices
# In[40]:


#Importing Pandas library
import pandas as pd


# In[41]:


#Importing data from csv file  
house_data=pd.read_csv('house-data.csv')


# In[42]:


#Modifying Data & Creating variables X_test ,Y_test ,X_train, Y_train
house_data=house_data.iloc[:, 2:]
X_train=house_data.iloc[:15000,1:]
Y_train=house_data.iloc[:15000,0]
X_test=house_data.iloc[15000:,1:]
Y_test=house_data.iloc[15000:,0]


# In[43]:


#Importing ExtraTreesRegressor & Mean Absolute Error Metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error as MAE


# In[44]:


#Making etr having ExtraTreesRegressor
etr=ExtraTreesRegressor()
etr.fit(X_train,Y_train)
predict=etr.predict(X_test)


# In[45]:


#Outputting Mean Absolute Error
print("Mean Absolute Error :",MAE(Y_test,predict))


# In[46]:


#DATA SET IS NOT MODIFIED TO WORK ACCURATELY WITH THE REGRESSOR
#It Just gives an idea about how to Implement the Models

