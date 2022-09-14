#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r'C:\work\learning\datasets\USA_Housing.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


sns.pairplot(df)


# In[8]:


sns.distplot(df['Price'])


# In[9]:


df.corr()


# In[10]:


sns.heatmap(df.corr(), annot=True) # pass number when true


# In[11]:


lst = list(df.columns)
lst.remove("Price")
print(lst)


# In[12]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]


# In[13]:


y = df['Price']


# In[14]:


# train test split data

from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


lm = LinearRegression()


# In[18]:


lm.fit(X_train, y_train)


# In[19]:


print(lm.intercept_)


# In[20]:


print(lm.coef_)


# In[21]:


X.columns


# In[22]:


cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])


# In[23]:


print(cdf)


# In[36]:


predicitions = lm.predict(X_test)


# In[37]:


predicitions # predicted price


# In[38]:


y_test


# In[39]:


plt.scatter(predicitions, y_test) # create a scatter plot to see the prediciton and the y_test


# In[40]:


sns.distplot((y_test - predicitions))


# In[41]:


from sklearn import metrics


# In[43]:


metrics.mean_absolute_error(y_test, predicitions)


# In[44]:


metrics.mean_squared_error(y_test, predicitions)


# In[45]:


np.sqrt(metrics.mean_squared_error(y_test, predicitions))


# In[ ]:




