
# coding: utf-8

# Random Forest Classifier On Dataset of Wharts.

# In[27]:


#Importing Pandas library
import pandas as pd


# In[28]:


#Importing data from csv file  
house_data=pd.read_csv('house-data.csv')


# In[29]:


#Modifying Data & Creating variables X_test ,Y_test ,X_train, Y_train
house_data=house_data.iloc[:, 2:]
X_train=house_data.iloc[:15000,1:]
Y_train=house_data.iloc[:15000,0]
X_test=house_data.iloc[15000:,1:]
Y_test=house_data.iloc[15000:,0]


# In[30]:


#Importing BaggingRegressor & Mean Absolute Error Metrics
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error as MAE


# In[31]:


#Making etc having BaggingRegressor
br=BaggingRegressor()
br.fit(X_train,Y_train)
predict=br.predict(X_test)


# In[32]:


#Outputting Mean Absolute Error
print("Mean Absolute Error :",MAE(Y_test,predict))


# In[ ]:


##DATA SET IS NOT MODIFIED TO WORK ACCURATELY WITH THE REGRESSOR

