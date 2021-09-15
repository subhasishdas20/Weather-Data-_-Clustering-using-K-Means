#!/usr/bin/env python
# coding: utf-8

# # Weather Data Clustering using K-Mean

# ### Importing Libaries 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ### Loading Dataset

# In[2]:


df=pd.read_csv(r"C:\Learning\python_class\minute_weather.csv")


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# ### Dropping Null Values

# In[7]:


df=df.dropna()


# In[8]:


df.isnull().sum()


# In[9]:


df= df.iloc[:,2:]


# In[10]:


df


# ### Scaling 
# 

# In[11]:


sc= StandardScaler()
df_scale=sc.fit_transform(df)


# In[12]:


df_scale


# ### Creating Model 

# In[13]:


from sklearn.cluster import KMeans
model = KMeans(7)
model=model.fit(df_scale)


# In[14]:


model.cluster_centers_


# In[15]:


model.labels_


# In[16]:


df["cluster"]=model.labels_


# In[17]:


df.head()


# In[18]:


df["cluster"].value_counts()


# In[19]:


model.inertia_


# ###Plotting

# In[21]:


df_elbow=[]
for i in range(1,15):
    model = KMeans(i)
    model.fit(df_scale)
    df_elbow.append(model.inertia_)
plt.plot(range(1,15), df_elbow)


# In[ ]:

