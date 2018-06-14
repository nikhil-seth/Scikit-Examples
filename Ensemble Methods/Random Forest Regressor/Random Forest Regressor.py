
# coding: utf-8

# Random Forest Regressor On Dataset of House Prices.

# In[33]:


#Importing Pandas library
import pandas as pd


# In[34]:


#Importing data from csv file  
house_data=pd.read_csv('house-data.csv')


# In[35]:


#Modifying Data & Creating variables X_test ,Y_test ,X_train, Y_train
house_data=house_data.iloc[:, 2:]
X_train=house_data.iloc[:15000,1:]
Y_train=house_data.iloc[:15000,0]
X_test=house_data.iloc[15000:,1:]
Y_test=house_data.iloc[15000:,0]


# In[36]:


#Importing RandomForestRegressor & Mean Absolute Error Metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as MAE


# In[37]:


#Making rfr having RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,Y_train)
predict=rfr.predict(X_test)


# In[38]:


#Outputting Mean Absolute Error
print("Mean Absolute Error :",MAE(Y_test,predict))


# In[39]:


#DATA SET IS NOT MODIFIED TO WORK ACCURATELY WITH THE REGRESSOR
#It Just gives an idea about how to Implement the Models

