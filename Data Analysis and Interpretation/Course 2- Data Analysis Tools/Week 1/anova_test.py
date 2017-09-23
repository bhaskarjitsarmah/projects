
# coding: utf-8

# ## Research Question

# The research question I am trying to answer in this assignment is **Is there any relationship between variables mpg (miles per gallon) of a car and cylinders (number of cylinders) the car have?** I have taken the dataset [auto-mpg dataset](http://archive.ics.uci.edu/ml/datasets/Auto+MPG) from UCI Machine Learning Repository.

# The variable **mpg** is my response variable which is continuous quantitative variable and the **cylinders** is the explanatory variable which is a categorical variable. So to test the relationship between these two variables, I need to perform the ANOVA Test.

# ## Loading libraries

# In[1]:

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi


# ## Loading the data

# In[2]:

data = pd.read_csv("auto-mpg.data", delim_whitespace=True, header=None, usecols=[0,1], names=['mpg', 'cylinders'], 
                   dtype={'mpg':np.float64, 'cylinders':'category'})


# ## Summarizing the data

# In[3]:

data.shape


# In[4]:

data.head(n=3)


# In[5]:

data.info()


# In[6]:

data.describe()


# In[7]:

# removing the unnecessary variable
data['cylinders'].value_counts()


# ## Setting Hypothesis

# As we are testing the relationship between the variables **mpg** and **cylinders**, we can explain the null hypothesis as follows-

# <ol><li>**Null Hypothesis:** There is nothing going on between the variables, there is no relationship between the two variables, in mathematical terms \begin{align}
# \\H_0: \mu_3 = \mu_4 = \mu_5 = \mu_6 = \mu_8 
# \end{align}</li><br><li>**Alternate Hypothesis:** There is something going on between the explanatory and response variable, or there is a relationship between the two, in mathematical terms \begin{align}
# \\H_A: \mu_3 \neq \mu_4 \neq \mu_5 \neq \mu_6 \neq \mu_8
# \end{align}</li></ol>

# ## Visualizing data

# In[8]:

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=data['cylinders'], y=data['mpg'], showmeans=True)
plt.show()


# From the above plot, it is clearly visible that the mean (triangular shape in red color) of the group with 8 number of cylinders does not overlap with another group means. We can now say that we have evidence against the null hypothesis and the variables are related to each other. But lets prove this by performing the ANOVA Test.

# ## ANOVA F Test

# In[9]:

# using ols function for calculating the F-statistic and associated p value
model = smf.ols(formula='mpg ~ cylinders', data=data)
results = model.fit()
print (results.summary())


# As we can observe from the above table, the F-statistic is very high at 172.6 with the very very low p-value. So, we can reject our null hypothesis and conclude that there is a relationship between the categorical predictor variable cylinder (number of cylinders in the car) and quantitative target variable mpg (mileage of the car).

# ## Post Hoc Analysis

# Now we have evidence against the null hypothesis and proved that there is a relationship between number of cylinders and miles per gallon of a car, we need to check which which number of cylinders effect mileage of car the most. In other words, mpg for which number of cylinders are significantly different from each other. To do this, we need to perform post-hoc analysis of the above ANOVA Test.

# In[10]:

mc1 = multi.MultiComparison(data['mpg'], data['cylinders'])
res1 = mc1.tukeyhsd()
print(res1.summary())


# From the above result, it is evident that the cars with 4 and 8 number of cylinders differs significantly in mileage (mpg) when compares to cars with 3, 5 and 6 cylinders.
