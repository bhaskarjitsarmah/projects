
# coding: utf-8

# ## Basic Linear Regression

# In this assignment, I have tried to build a basic linear regression model. I have taken the [Auto MPG Data Set](http://archive.ics.uci.edu/ml/datasets/Auto+MPG) from UCI Machine Learning Repository. The response variable is **mpg** (miles per gallon) and explanatory variable is **weight** of the car. Both the variable are numeric in nature.

# ## Loading Libraries

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels.formula.api as smf


# ## Loading Data Set

# In[2]:

data = pd.read_csv("auto-mpg.data", 
                   delim_whitespace=True, 
                   header=None, 
                   usecols=[0, 4], 
                   names=['mpg', 'weight'])


# ## Summarizing Data Set

# In[3]:

data.describe()


# ## Centering Explanatory Variable

# As we can see from above summary, our explanatory variable **weight** does not have 0 in its range. So we need to do **centering** for the variable to get some meaningful results.

# In[4]:

mean = round(data['weight'].mean(), 2)
data['weight'] = data['weight'] - mean
data.describe()


# ## Linear Regression

# In[5]:

scat1 = sns.regplot(x="weight", y="mpg", scatter=True, data=data)
plt.xlabel('Weight of the Car')
plt.ylabel('Miles Per Gallon')
plt.title ('Scatterplot for the Association Between Car Weight and Mileage of the Car')
plt.show()


# In[6]:

print ("OLS regression model for the association between car weight and mileage of the car")
reg1 = smf.ols('mpg ~ weight', data=data).fit()
print (reg1.summary())


# From the above simple linear regression model, we can make the following conclusions <ol>
# <li>We got **p-value = 2.97e-103** which is very very less. So we can say that our basic linear regression model is significant.</li>
# <li>The **R-squared = 0.692**, which means that 69.2% of the variable in the response variable **mpg** is explained by the explanatory variable **weight**</li>
# <li>The Regression Equation is given by <br>**=> mpg = intercept - 0.0077 * weight**<br>**=> mpg = 23.5146 - 0.0077 * weight**</li></ol> 
