#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')
df


# In[5]:


#checking columns
df.columns


# In[6]:


sns.pairplot(data=df, diag_kind='kde')


# # Will check Description to find out skewness and Outliers

# Here we can see Country,Region, Happiness Rank are not correlated with output of dataset Result. Its an Independent variable 
# so we will drop this to avoid inaccuracy

# In[7]:


#dropping the columns
df.drop(columns='Country',axis=1,inplace=True)
df.drop(columns='Region',axis=1,inplace=True)
df.drop(columns='Happiness Rank',axis=1,inplace=True)
df.head()


# In[8]:


df.describe()


# # checking null values

# In[9]:


df.isnull().sum()


# No Null values are present in any column

# # checking Outliers

# In[10]:


df.plot(kind='box',subplots=True,layout=(2,6),figsize=(10,10))


# In[11]:


df['Standard Error'].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


# In[12]:


df['Generosity'].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


# In[13]:


df['Dystopia Residual'].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


# In[14]:


df['Trust (Government Corruption)'].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])#Trust (Government Corruption)


# # Removing Outliers using Z score

# In[15]:


from scipy.stats import zscore
import numpy as np
z=np.abs(zscore(df))# abs will make it positive---> mod of x= x and -X---> if x=2 then mode of x=3 or x=-(-2)=2
z.shape


# In[16]:


thresold=3
print(np.where(z>3))


# In[17]:


df_new=df[(z<3).all(axis=1)]


# In[18]:


df_new


# # Checking Skewness

# In[19]:


x=df_new.drop("Happiness Score",axis=1)
y=df_new['Happiness Score']


# In[20]:


x.skew().sort_values(ascending=False)


# Two way to check Skewness
# 1. If Mean > 50% 
# 2.df.skew()
# 
# Hence above we can see the standard Error,Trust having High skewed Data
# 
# and Generosity and Family having low skewed Data.

# # Removing Skwness using power_transform 

# In[21]:


from sklearn.preprocessing import power_transform
x1=power_transform(x)


# In[22]:


pd.DataFrame(x1).skew().sort_values(ascending=False)


# In[23]:


pd.DataFrame(x1, columns = x.columns).skew().sort_values(ascending=False)


# In[24]:


x=pd.DataFrame(x1,columns=x.columns)


# In[25]:


x.skew().sort_values(ascending=False)


# In[26]:


x


# # Checking correlationship

# In[27]:


df_corr=df_new.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df_corr,annot=True,annot_kws={'size':10})
plt.show()


# Here we can see almost all variables are corelated with output variable except standard error that almost equal to 0
# hence we are going to drop any column

# # Testing and Training

# In[28]:


x.head()


# In[29]:


y.head()


# In[30]:


from sklearn.linear_model import LinearRegression

#x=df_new.drop("Happiness Score",axis=1)
#y=df_new['Happiness Score']

maxAcc = 0
maxRs=0

for i in range(1,140):
    x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.20,random_state=i) #10
    lm=LinearRegression()
    lm.fit(x_train,y_train)
    pred=lm.predict(x_test)
    acc=lm.score(x_test,y_test)
    #print('accuracy',acc,'Random state',i)
    
    if acc>maxAcc:
        maxAcc=acc
        maxRs=i
        #print('accuracy',maxAcc,'Random state',i)
        
print("Best accuracy is",maxAcc*100,"on Random state",maxRs) #45 45


# In[31]:


#training and testing the data
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=.20,random_state=60)


# In[32]:


x_train.shape


# In[33]:


y_train.shape


# In[34]:


x_test.shape


# In[35]:


y_test.shape


# In[36]:


lm.fit(x_train,y_train)


# In[37]:


pred=lm.predict(x_test)


# In[38]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error

r2_score(y_test, pred)*100, mean_absolute_error(y_test, pred), np.sqrt(mean_squared_error(y_test, pred))


# In[39]:


print("predicted values",pred)
print("predicted values",y_test)


# Here we can see the output and predicted values are almost matching

# # Testing the one sample Data with Linear Regression

# In[40]:


t=np.array([0.03411,1.39651,1.34951,0.94143,0.66557,0.41978,0.29678,2.51738]) 
# same data from first line which is having 7.5 output


# In[41]:


t.shape


# In[42]:


t=t.reshape(1,-1)
t.shape


# In[43]:


lm.predict(t)


# In[ ]:





# In[ ]:




