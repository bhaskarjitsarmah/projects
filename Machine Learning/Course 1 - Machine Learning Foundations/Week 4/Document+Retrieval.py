
# coding: utf-8

# # Document retrieval from wikipedia data

# ## Fire up GraphLab Create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[1]:

import graphlab


# # Load some text data - from wikipedia, pages on people

# In[2]:

people = graphlab.SFrame('people_wiki.gl/')


# Data contains:  link to wikipedia article, name of person, text of article.

# In[3]:

people.head()


# In[4]:

len(people)


# # Explore the dataset and checkout the text it contains
# 
# ## Exploring the entry for Elton John

# In[5]:

elton = people[people['name'] == 'Elton John']


# In[6]:

elton


# In[7]:

elton['text']


# ## Exploring the entry for actor Narendra Modi

# In[8]:

Modi = people[people['name'] == 'Narendra Modi']
Modi['text']


# # Get the word counts for Elton John article

# In[9]:

elton['word_count'] = graphlab.text_analytics.count_words(elton['text'])


# In[10]:

print elton['word_count']


# ## Sort the word counts for the Elton John article

# ### Turning dictonary of word counts into a table

# In[11]:

elton_word_count_table = elton[['word_count']].stack('word_count', new_column_name = ['word','count'])


# ### Sorting the word counts to show most common words at the top

# In[12]:

elton_word_count_table.head()


# In[13]:

elton_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# # Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[14]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[15]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf


# In[16]:

people['tfidf'] = tfidf


# ## Examine the TF-IDF for the Elton John article

# In[17]:

elton = people[people['name'] == 'Elton John']


# In[18]:

elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# Words with highest TF-IDF are much more informative.

# # Manually compute distances between a few people
# 
# Let's manually compare the distances between the articles for a few famous people.  

# In[19]:

victoria = people[people['name'] == 'Victoria Beckham']


# In[20]:

paul = people[people['name'] == 'Paul McCartney']


# ## Is Elton John closer to Victoria Beckham than to Paul McCartney?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about president Obama is closer to the one about former president Clinton than that of footballer David Beckham.

# In[21]:

graphlab.distances.cosine(elton['tfidf'][0], victoria['tfidf'][0])


# In[22]:

graphlab.distances.cosine(elton['tfidf'][0], paul['tfidf'][0])


# # Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  

# In[23]:

model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# # Applying the nearest-neighbors model for retrieval

# ## Celebrities closest to Elton John

# In[24]:

model.query(elton, verbose=False)

