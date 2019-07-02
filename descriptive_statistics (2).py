
# coding: utf-8

# In[1]:


import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


# In[2]:


data = pd.read_csv("Z://sem7//StudentsPerformance.csv") 
rs=data.head()
rs


# In[3]:


mean1 = data['math_score'].mean()


# In[4]:


print ('Mean : ' + str(mean1))


# In[5]:


slice = data.iloc[:,[5, 6]]


# In[39]:


bp = slice.boxplot(column='math_score', by='reading_score')
axes = pl.gca()
axes.set_xlim([0,50])
axes.set_ylim([0,50])
pl.show()


# In[7]:


import scipy
from scipy import stats
scipy.stats.hmean(data.loc[:,"math_score"])


# In[10]:


score=data.math_score
score.mean()


# In[11]:


scipy.stats.mstats.gmean(score, axis=0)


# In[12]:


scipy.stats.iqr(data.reading_score)


# In[27]:



pl.boxplot(data.math_score)


# In[32]:


import seaborn as sns
his=sns.countplot(x="reading_score",data=data.sample(30))


# In[33]:


m=data.math_score.sample(30)
n=data.reading_score.sample(30)
plt.pie(n,labels=m,autopct='%.1f%%',startangle=90)
plt.show()


# In[37]:


pl.bar(data.math_score.sample(30),data.reading_score.sample(30))
pl.show()


# In[38]:


import seaborn as sns
his=sns.countplot(x="math_score",data=data.sample(30))


# In[46]:


sns.boxplot(x="gender",y="math_score",data=data)


# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


# In[42]:


x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black');


# In[51]:


import matplotlib.pyplot as plt

x = data.writing_score
y = data.reading_score

pl.scatter(x,y, label='outlier', color='k', s=10, marker="o")

pl.xlabel('x')
pl.ylabel('y')
pl.title('Scatter plot for math and reading score')
pl.legend()
pl.show()

