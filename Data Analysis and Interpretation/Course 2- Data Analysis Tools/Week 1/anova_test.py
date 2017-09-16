
# coding: utf-8

# ## Research Question

# Is there any relationship between variables **diet** taken and **weight loss** by a respondent? I have taken the dataset **diet_exercise** to perform this test.

# The dataset consist of two explanatory variables **Exercise** and **Diet** (Categorical in nature) and the response variable **WeightLoss**. Since the explanatory variable is categorical in nature and **WeightLoss** variable is numerical in nature, we need to perform the **ANOVA** test to check for any relationship between these two variables. And we can ignore the variable **Exercise** as of now as we are only interested in knowing the relationship between the variables **Diet** and **WeightLoss**.

# ## Loading libraries

# In[1]:

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi


# ## Loading the data

# In[2]:

data = pd.read_csv("diet_exercise.csv")


# In[3]:

data.shape


# In[4]:

data.head(n=3)


# In[5]:

# removing the unnecessary variable
data = data.drop('Exercise', axis=1)


# ## Setting Hypothesis

# As we are testing the relationship between the variables **Diet** and **WeightLoss**, we can explain the null hypothesis as follows-

# <ol><li>**Null Hypothesis:** There is nothing going on between the variables, there is no relationship between the two variables, in mathematical terms \begin{align}
# \\H_0: \mu_A & = \mu_B 
# \end{align}</li><br><li>**Alternate Hypothesis:** There is something going on between the explanatory and response variable, or there is a relationship between the two, in mathematical terms \begin{align}
# \\H_A: \mu_A & \neq \mu_B 
# \end{align}</li></ol>

# ## Visualizing data

# In[8]:

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=data['Diet'], y=data['WeightLoss'])
plt.show()


# From the above plot it looks like the mean of these two groups of observations (Diet A and Diet B) are different from each other. It signifies the distributions for these two Diet categories are from two independent populations and there is a relationship between the variable **Diet** and **WeightLoss**. But we need to prove this visual inference through a statistical test.

# In[19]:

# using ols function for calculating the F-statistic and associated p value
model1 = smf.ols(formula='WeightLoss ~ C(Diet)', data=data)
results1 = model1.fit()
print (results1.summary())


# From the statistical results above we get **p-value = 0.00133** which is less than the standard significance level 0.05. Therefore we can reject our null hypothesis as the data shows significant evidence that probability of finding the observed or more extreme results provided the null hypothesis is true is very very less. So the relationship between the variable **Diet** and **WeightLoss** is statistically significant.
