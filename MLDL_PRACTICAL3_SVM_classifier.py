#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Import libraries
import numpy as np
import pandas as pd


# In[22]:


#load datasets
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


# In[24]:


data.data


# In[25]:


data.feature_names


# In[26]:


data.target


# In[27]:


data.target_names


# In[23]:


#create dtaframe
df = pd.DataFrame(np.c_[data.data, data.target], columns=[list(data.feature_names)+['target']])
df.head()


# In[28]:


df.head()


# In[30]:


df.tail()


# In[31]:


df.shape


# In[32]:


#shape the data
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]


# In[33]:





# In[34]:


y


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
 
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)


# In[53]:


## SVC with Kernel Linear
from sklearn.svm import SVC
 
classification_linear = SVC(kernel='linear')
classification_linear.fit(X_train, y_train)
 
classification_linear.score(X_test, y_test)


# In[60]:


## SVC with Kernel Linear by changing degree
from sklearn.svm import SVC
 
classification_linear = SVC(kernel='linear' , degree = 2)
classification_linear.fit(X_train, y_train)
 
classification_linear.score(X_test, y_test)


# In[54]:


# Feature Scaling
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
 
sc.fit(X_train)
 
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

 


# In[55]:


#After Appling Feature Scalling
classification_linear_2 = SVC(kernel='linear')
classification_linear_2.fit(X_train_sc, y_train)
 
classification_linear_2.score(X_test_sc, y_test)


# In[61]:


#After Appling Feature Scalling and change degree
classification_linear_2 = SVC(kernel='linear' , degree = 2)
classification_linear_2.fit(X_train_sc, y_train)
 
classification_linear_2.score(X_test_sc, y_test)


# In[64]:


## SVC with kernel Polynomial
 
classification_poly = SVC(kernel='poly', degree=2)
classification_poly.fit(X_train_sc, y_train)
 
classification_poly.score(X_test_sc, y_test)


# In[44]:


## Train Support Vector Classification Model
 
from sklearn.svm import SVC
 
classification_rbf = SVC(kernel='rbf')
classification_rbf.fit(X_train, y_train)
 
classification_rbf.score(X_test, y_test)


# In[52]:


#After Appling Feature Scalling
classification_rbf_2 = SVC(kernel='rbf')
classification_rbf_2.fit(X_train_sc, y_train)
 
classification_rbf_2.score(X_test_sc, y_test)


# In[65]:


## Predict Cancer
 
patient1 = [17.99,
 10.38,
 122.8,
 1001.0,
 0.1184,
 0.2776,
 0.3001,
 0.1471,
 0.2419,
 0.07871,
 1.095,
 0.9053,
 8.589,
 153.4,
 0.006399,
 0.04904,
 0.05373,
 0.01587,
 0.03003,
 0.006193,
 25.38,
 17.33,
 184.6,
 2019.0,
 0.1622,
 0.6656,
 0.7119,
 0.2654,
 0.4601,
 0.1189]


# In[66]:


patient1_sc = sc.transform(np.array([patient1]))
patient1_sc


# In[73]:


pred= classification_linear.predict(patient1_sc)
pred


# In[74]:


data.target_names


# In[75]:


if pred[0] == 0:
  print('Patient has Cancer (malignant tumor)')
else:
  print('Patient has no Cancer (malignant benign)')


# In[ ]:




