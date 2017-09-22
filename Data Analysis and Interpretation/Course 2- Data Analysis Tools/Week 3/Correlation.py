
# coding: utf-8

# ## Research Question

# In this assignment, I have tried to find out the answer for the research question: **Is there a relationship between the sepal length and petal length of flower?** I have taken this dataset from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Iris). Here both the variables are quantitative in nature, below images shows sepals and petals of a flower.

# ![alt](sepal.png) | ![alt](petal.jpg)

# ## Loading Libraries

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st


# ## Loading Data

# In[2]:

data = pd.read_csv("iris.data", header=None, usecols=[0,2], names=['sepal_length', 'petal_length'])


# In[3]:

data.head()


# In[4]:

data.info()


# ## Visualizing Data

# In[5]:

scat1 = sns.regplot(x="sepal_length", y="petal_length", fit_reg=True, data=data)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Scatterplot for the Association Between Sepal Length and Petal Length')
plt.show()


# From the above plot, it is clearly evident that there is a very strong positive linear relationship between the variables **Sepal Length** and **Petal Length**. Let's now compute the value of correlation cofficient **r** using the library scipy.stats

# In[6]:

print ('association between sepal length and petal length')
print (st.pearsonr(data['sepal_length'], data['petal_length']))


# After computing, we get the r=0.87175415730487116, this means there is a very strong association between the variables and r^2 = 0.759955311 means that 76% variation in the variable **Petal Length** is explained by the explanatory variable **Sepal Length**. r^2 is also known as coefficient of determination
