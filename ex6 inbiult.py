
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

X = [[1,1],[2,1],[2,2],[3,2]]
X=np.array(X)
plt.scatter(X[:,0],X[:,1])
plt.show()


# In[2]:


euc_dis_original=euclidean_distances(X, X)
print(euc_dis_original)


# In[4]:


mds = manifold.MDS(n_components=2, dissimilarity="euclidean", n_init=100, max_iter=1000, random_state=10)
results = mds.fit(X)


# In[5]:


coords = results.embedding_
euc_dis_mod=euclidean_distances(coords, coords)
print(euc_dis_mod)
fig = plt.figure(figsize=(12,10))

plt.subplots_adjust(bottom = 0.1)
plt.scatter(coords[:, 0], coords[:, 1])


plt.show()


# In[6]:


plt.scatter(euc_dis_original, euc_dis_mod)
plt.show()


# In[7]:


plt.plot(euc_dis_original, euc_dis_mod)
plt.show()


# In[8]:


print(euc_dis_original-euc_dis_mod)

