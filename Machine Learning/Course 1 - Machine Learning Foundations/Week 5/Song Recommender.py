
# coding: utf-8

# # Building a song recommender
# 
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[1]:

import graphlab


# # Load music data

# In[2]:

song_data = graphlab.SFrame('song_data.gl/')


# # Explore data
# 
# Music data shows how many times a user listened to a song, as well as the details of the song.

# In[3]:

song_data.head()


# ## Showing the most popular songs in the dataset

# In[4]:

graphlab.canvas.set_target('ipynb')


# In[5]:

song_data['song'].show()


# In[6]:

len(song_data)


# ## Count number of unique users in the dataset

# In[7]:

users = song_data['user_id'].unique()


# In[8]:

len(users)


# In[41]:

song_data.groupby(key_columns='artist', operations={'unique_users': graphlab.aggregate.COUNT_DISTINCT('user_id')}).sort(
    ['unique_users'])


# In[39]:

data[(data['artist'] == "Kanye West") | (data['artist'] == "Lady GaGa") | (data['artist'] == "Foo Fighters")
     | (data['artist'] == "Taylor Swift")]


# # Create a song recommender

# In[9]:

train_data,test_data = song_data.random_split(.8,seed=0)


# ## Simple popularity-based recommender

# In[10]:

popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id='user_id',
                                                         item_id='song')


# ### Use the popularity model to make some predictions
# 
# A popularity model makes the same prediction for all users, so provides no personalization.

# In[11]:

popularity_model.recommend(users=[users[0]])


# In[12]:

popularity_model.recommend(users=[users[1]])


# ## Build a song recommender with personalization
# 
# We now create a model that allows us to make personalized recommendations to each user. 

# In[13]:

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')


# ### Applying the personalized model to make song recommendations
# 
# As you can see, different users get different recommendations now.

# In[14]:

personalized_model.recommend(users=[users[0]])


# In[15]:

personalized_model.recommend(users=[users[1]])


# ### We can also apply the model to find similar songs to any song in the dataset

# In[16]:

personalized_model.get_similar_items(['With Or Without You - U2'])


# In[17]:

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])


# # Quantitative comparison between the models
# 
# We now formally compare the popularity and the personalized models using precision-recall curves. 

# In[18]:

if graphlab.version[:3] >= "1.6":
    model_performance = graphlab.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
    graphlab.show_comparison(model_performance,[popularity_model, personalized_model])
else:
    get_ipython().magic(u'matplotlib inline')
    model_performance = graphlab.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample=.05)


# The curve shows that the personalized model provides much better performance. 
