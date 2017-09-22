
# coding: utf-8

# ## Research Question

# Is there a moderation effect of number of cylinders on the relationship between mileage (miles per gallon, mpg) and horsepower of an automobile? To answer this question I have taken the [auto-mpg dataset](http://archive.ics.uci.edu/ml/datasets/Auto+MPG) from UCI Machine Learning Repository. Here the response variable is **mpg** (miles per gallon), explanatory variable is **horsepower** and moderator variable is **cylinders** (number of cylinders present in the car)

# ## Loading Libraries

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st


# ## Loading Dataset

# In[2]:

data = pd.read_csv("auto-mpg.data", delim_whitespace=True, header=None, usecols=[0,1,3], names=['mpg', 'cylinders', 'horsepower'], 

                   dtype={'mpg':np.float64, 'cylinders':'category'})

data['horsepower'] = data['horsepower'].convert_objects(convert_numeric=True)
#data['horsepower'] = pd.to_numeric(data['horsepower'])
data = data.dropna()


# ## Summarizing Dataset

# In[3]:

data.head()


# In[4]:

data.info()


# In[5]:

data.describe()


# In[6]:

data['cylinders'].value_counts()


# ## Relationship Between Horse Power and Miles Per Gallon

# In[7]:

scat1 = sns.regplot(x="horsepower", y="mpg", fit_reg=True, data=data)
plt.xlabel('Horse Power')
plt.ylabel('Miles Per Gallon')
plt.title('Scatterplot for the Association Between Horse Power and Miles Per Gallon')
plt.show()


# In[8]:

print ('association between sepal length and petal length')
print (st.pearsonr(data['horsepower'], data['mpg']))


# As it is seen from both the scatterplot and above pearson correlation coefficient, there is a very strong negative association between the variables **horsepower** and **mpg**. The correlation cofficient is r=-0.77842678389777586 with p-value=7.031989029404564e-81 (which is very very small). So we can conclude that the test is statistically significant and proves that there is a negative association between these two variables.

# ## Moderation Effect of Number of Cylinders

# Now we will check moderation effect of the variable **cylinders** on the above association between the variables **horsepower** and **mpg**. We will check whether the negative association the variables still holds true for all different levels of the moderation variable or not? If yes then we can say there is no moderation effect, and if the answer is no then we can conclude that there is a moderation effect of the variable **cylinders** on the above negative association between **horsepower** and **mpg**

# In[9]:

data['cylinders'] = data['cylinders'].astype('int')

data_3 = data[data['cylinders'] == 3]
data_4 = data[data['cylinders'] == 4]
data_5 = data[data['cylinders'] == 5]
data_6 = data[data['cylinders'] == 6]
data_8 = data[data['cylinders'] == 8]


# In[10]:

print ('association between horsepower and mpg for 3 cylinders cars')
print (st.pearsonr(data_3['horsepower'], data_3['mpg']))
print ('       ')
print ('association between horsepower and mpg for 4 cylinders cars')
print (st.pearsonr(data_4['horsepower'], data_4['mpg']))
print ('       ')
print ('association between horsepower and mpg for 5 cylinders cars')
print (st.pearsonr(data_5['horsepower'], data_5['mpg']))
print ('       ')
print ('association between horsepower and mpg for 6 cylinders cars')
print (st.pearsonr(data_6['horsepower'], data_6['mpg']))
print ('       ')
print ('association between horsepower and mpg for 8 cylinders cars')
print (st.pearsonr(data_8['horsepower'], data_8['mpg']))


# From the above coefficients, we can see that results for data with only 4 and 8 number of cylinders are significant. While the negative association between **horsepower** and **mpg** does not hold true for the data with 3, 5 and 6 number of cylinders. Let's also plot these data for all different levels of the variable **cylinder** to visually inspect the relationship.

# In[11]:

scat1 = sns.regplot(x="horsepower", y="mpg", data=data_3)
plt.xlabel('Engine Horse Power')
plt.ylabel('MPG')
plt.title('Scatterplot for the Association Between Horsepower and MPG Rate for 3 Cylinder Cars')
plt.show()


# In[12]:

scat2 = sns.regplot(x="horsepower", y="mpg", data=data_4)
plt.xlabel('Engine Horse Power')
plt.ylabel('MPG')
plt.title('Scatterplot for the Association Between Horsepower and MPG Rate for 4 Cylinder Cars')
plt.show()


# In[13]:

scat3 = sns.regplot(x="horsepower", y="mpg", data=data_5)
plt.xlabel('Engine Horse Power')
plt.ylabel('MPG')
plt.title('Scatterplot for the Association Between Horsepower and MPG Rate for 5 Cylinder Cars')
plt.show()


# In[14]:

scat4 = sns.regplot(x="horsepower", y="mpg", data=data_6)
plt.xlabel('Engine Horse Power')
plt.ylabel('MPG')
plt.title('Scatterplot for the Association Between Horsepower and MPG Rate for 6 Cylinder Cars')
plt.show()


# In[15]:

scat5 = sns.regplot(x="horsepower", y="mpg", data=data_8)
plt.xlabel('Engine Horse Power')
plt.ylabel('MPG')
plt.title('Scatterplot for the Association Between Horsepower and MPG Rate for 8 Cylinder Cars')
plt.show()


# So, from the above plots we can conclude that though there is a negative association between **horsepower** and **mpg** for the whole dataset. There is a moderation effect of number of cylinders on this negative association. This relationship does not hold true for the data points with 3, 5 and 6 number of cylinders. 
