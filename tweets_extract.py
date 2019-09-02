#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tweepy 


# In[30]:


consumer_key = "Yf1DrVA0XG1y8wwFcz49AP1Sj" 
consumer_secret = "f1kxVI4zSmB4pPv0PObgkYR34G2FAyh3OxszxmpFY5iLI8BrnE"
access_key = "1138319595528380416-QtgahgmXizpd2rabhwdz5AuPB5lDn4"
access_secret = "BJSPgaYQcFNibeYXQxnkzS2wNqPb8a1GDmLq2fcGQca3i "


# In[31]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth) 


# In[32]:



public_tweets = api.home_timeline()

for tweet in public_tweets:
   
   print (tweet.text)


# In[33]:


print (tweet.created_at)


# In[34]:


print( tweet.user.screen_name)


# In[35]:


api = tweepy.API(auth)
name = "nytimes"
tweetCount = 20
results = api.user_timeline(id=name, count=tweetCount)
for tweet in results:

   print (tweet.text)


# In[36]:


api = tweepy.API(auth)
name = "nytimes"
tweetCount = 20
results = api.user_timeline(id=name, count=tweetCount)
for tweet in results:
 
   print (tweet.text)


# In[ ]:




