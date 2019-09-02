
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import copy

X = [[1,1],[2,1],[2,2],[3,2]]
X=np.array(X)


plt.scatter(X[:,0], X[:,1])
plt.title('Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[3]:


Xmean = np.mean(X[:,0])
Ymean = np.mean(X[:,1])

msubX = X[:,0] - Xmean

msubY = X[:,1] - Ymean


# In[4]:


msubData = np.column_stack((msubX,msubY))

msubData


# In[5]:


plt.scatter(msubData[:,0], msubData[:,1])
plt.title('Mean SubtractedDataset')
plt.xlabel('X - X_Mean')
plt.ylabel('Y - Y_Mean')
plt.show()


# In[6]:


VarX = 0

for ele in msubData[:,0]:
    VarX = VarX + ele*ele
    
VarX = VarX/3

VarX


# In[7]:


VarY = 0 

for ele in msubData[:,1]:
    VarY = VarY + ele*ele

VarY = VarY/3

VarY


# In[8]:


VarXY = 0
for ele,ele1 in msubData[:]:
    
    VarXY = VarXY + ele*ele1
    
VarXY = VarXY/3

VarXY


# In[9]:


Cmat = np.column_stack(([VarX,VarXY],[VarXY,VarY]))

Cmat


# In[10]:


Cmatprod = np.dot(msubData,msubData.T)

Cmatprod


# In[13]:


eigenval, eigenvec = np.linalg.eig(Cmatprod)

print(eigenval)

print(eigenvec)


# In[12]:


evec=eigenvec[:,[0]]
print(evec)


# In[14]:


e_val=eigenval[0]
print(e_val)


# In[15]:


import math
mdsproj=math.sqrt(e_val)*evec
print(mdsproj)


# In[16]:


from sklearn.metrics.pairwise import euclidean_distances
euc_dis_original=euclidean_distances(msubData, msubData)
print(euc_dis_original)


# In[17]:


euc_dis_mod=euclidean_distances(mdsproj, mdsproj)
print(euc_dis_mod)


# In[18]:


plt.scatter(euc_dis_original, euc_dis_mod)
plt.show()


# In[19]:


plt.plot(euc_dis_original, euc_dis_mod)
plt.show()


# In[20]:


print(euc_dis_original-euc_dis_mod)

