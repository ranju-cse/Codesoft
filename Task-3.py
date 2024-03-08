#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn .model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import warnings
warnings.simplefilter("ignore")


# # Loading Dataset

# In[2]:


df=pd.read_csv("IMDB-Movie-Data.csv")


# In[3]:


df.head(5)


# In[4]:


df.tail(5)


# In[5]:


df.keys()


# In[6]:


col ="Votes"
df[col].hist()
plt.suptitle(col)
plt.show()


# In[7]:


col ="Revenue (Millions)"
df[col].hist()
plt.suptitle(col)
plt.show()


# In[8]:


pd.DataFrame(df)


# # Finding Shapes in Our Dataset

# In[9]:


df.shape


# In[10]:


print("Number of rows",df.shape[0])
print("Number of column",df.shape[1])


# # Getting Information about our dataset

# In[11]:


df.info()


# # Checking Missing Values

# In[12]:


df.isnull().values.any()


# In[13]:


df.isnull().sum()


# In[14]:


sns.heatmap(df.isnull())


# In[15]:


miss_val=df.isnull().sum()* 100/len(df)


# In[16]:


print(miss_val)


# # Drop the missing values

# In[17]:


df.dropna(axis=0)


# # Checking the Duplicate Values

# In[18]:


df_dup=df.duplicated().any()


# In[19]:


print(df_dup)  


# In[20]:


df=df.drop_duplicates()
df


# In[21]:


df.describe()


# # Displaying the movie having runtime>=180 Minutes

# In[22]:


df.columns


# In[23]:


df[df['Runtime (Minutes)']>=180]['Title']


# # Highest average voting year

# In[24]:


df.groupby('Year')['Votes'].mean().sort_values(ascending=False)


# In[25]:


plt.scatter(df['Year'],df['Votes'])
plt.xlabel('Year')
plt.ylabel('Votes')
plt.title('votes by year')
plt.show()


# In[26]:


plt.bar(df['Year'],df['Votes'])
plt.xlabel('Year')
plt.ylabel('Votes')
plt.title('votes by year')
plt.show()


# In[27]:


sns.countplot(x='Year', data=df)
plt.show()


# In[28]:


sns.barplot(x=df['Year'], y=df['Votes'])
plt.title('Votes by Year')
plt.xlabel('Year')
plt.ylabel('Votes')
plt.show()


# # Highest Average Revenue

# In[29]:


df.columns


# In[30]:


df.groupby('Year')['Revenue (Millions)'].mean().sort_values(ascending=False)


# In[31]:


sns.barplot(x=df['Year'], y=df['Revenue (Millions)'])
plt.title('Votes by Year')
plt.xlabel('Year')
plt.ylabel('Revenue (Millions)')
plt.show()


# # Average Rating Of Each Director

# In[32]:


df.groupby('Director')['Rating'].mean().sort_values(ascending=False)


# # Top 10 Lengthy Movies

# In[33]:


df.columns


# In[34]:


top10_len=df.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']].set_index('Title')


# In[35]:


top10_len


# In[36]:


top10_len.reset_index(inplace=True)
sns.barplot(x='Runtime (Minutes)', y='Title', data=top10_len)
plt.title('Top 10 Movies by Runtime')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Movie Title')
plt.show()


# # Most Popular Movies

# In[37]:


df.columns


# In[38]:


top10_len=df.nlargest(10,'Rating')[['Title','Rating','Director']].set_index('Title')


# In[39]:


plt.figure(figsize=(10, 6))
sns.countplot(y='Director', data=top10_len)
plt.title('Count of Movies Directed by Directors in Top 10')
plt.xlabel('Count')
plt.ylabel('Director')
plt.show()


# In[ ]:




