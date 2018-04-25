
# coding: utf-8

# # Find duplicate questions on StackOverflow by their embeddings
# 
# In this assignment you will learn how to calculate a similarity for pieces of text. Using this approach you will know how to find duplicate questions from [StackOverflow](https://stackoverflow.com).

# ### Libraries
# 
# In this task you will you will need the following libraries:
# - [StarSpace](https://github.com/facebookresearch/StarSpace) — a general-purpose model for efficient learning of entity embeddings from Facebook
# - [Gensim](https://radimrehurek.com/gensim/) — a tool for solving various NLP-related tasks (topic modeling, text representation, ...)
# - [Numpy](http://www.numpy.org) — a package for scientific computing.
# - [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
# - [Nltk](http://www.nltk.org) — a platform to work with human language data.

# ### Data
# 
# The following cell will download all data required for this assignment into the folder `week3/data`.

# In[2]:


import sys
sys.path.append("..")
from common.download_utils import download_week3_resources

download_week3_resources()


# ### Grading
# We will create a grader instace below and use it to collect your answers. Note that these outputs will be stored locally inside grader and will be uploaded to platform only after running submiting function in the last part of this assignment. If you want to make partial submission, you can run that cell any time you want.

# In[3]:


from grader import Grader


# In[4]:


grader = Grader()


# ## Word embedding
# 
# To solve the problem, you will use two different models of embeddings:
# 
#  - [Pre-trained word vectors](https://code.google.com/archive/p/word2vec/) from Google which were trained on a part of Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. You need to download it by following this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
#  - Representations using StarSpace on StackOverflow data sample. You will need to train them from scratch.

# It's always easier to start with pre-trained embeddings. Unpack the pre-trained Goggle's vectors and upload them using the function [KeyedVectors.load_word2vec_format](https://radimrehurek.com/gensim/models/keyedvectors.html) from gensim library with the parameter *binary=True*. If the size of the embeddings is larger than the avaliable memory, you could load only a part of the embeddings by defining the parameter *limit* (recommended: 500000).

# In[5]:


from gensim.models.keyedvectors import KeyedVectors


# In[6]:


wv_embeddings = KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)


# ### How to work with Google's word2vec embeddings?
# 
# Once you have loaded the representations, make sure you can access them. First, you can check if the loaded embeddings contain a word:
#     
#     'word' in wv_embeddings
#     
# Second, to get the corresponding embedding you can use the square brackets:
# 
#     wv_embeddings['word']
#  
# ### Checking that the embeddings are correct 
#  
# To prevent any errors during the first stage, we can check that the loaded embeddings are correct. You can call the function *check_embeddings*, implemented below, which runs 3 tests:
# 1. Find the most similar word for provided "positive" and "negative" words.
# 2. Find which word from the given list doesn’t go with the others.
# 3. Find the most similar word for the provided one.
# 
# In the right case the function will return the string *These embeddings look good*. Othervise, you need to validate the previous steps.

# In[7]:


def check_embeddings(embeddings):
    error_text = "Something wrong with your embeddings ('%s test isn't correct)."
    most_similar = embeddings.most_similar(positive=['woman', 'king'], negative=['man'])
    if len(most_similar) < 1 or most_similar[0][0] != 'queen':
        return error_text % "Most similar"

    doesnt_match = embeddings.doesnt_match(['breakfast', 'cereal', 'dinner', 'lunch'])
    if doesnt_match != 'cereal':
        return error_text % "Doesn't match"
    
    most_similar_to_given = embeddings.most_similar_to_given('music', ['water', 'sound', 'backpack', 'mouse'])
    if most_similar_to_given != 'sound':
        return error_text % "Most similar to given"
    
    return "These embeddings look good."


# In[8]:


print(check_embeddings(wv_embeddings))


# ## From word to text embeddings
# 
# **Task 1 (Question2Vec).** Usually, we have word-based embeddings, but for the task we need to create a representation for the whole question. It could be done in different ways. In our case we will use a **mean** of all word vectors in the question. Now you need to implement the function *question_to_vec*, which calculates the question representation described above.
# 
# Note that there could be words without the corresponding embeddings. In this case, you can just skip these words and don't take them into account during calculating the result. If the question doesn't contain any known word with embedding, the function should return a zero vector.

# In[9]:


import numpy as np


# In[13]:


def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    res = np.zeros(dim)
    n = 0
    for word in question.split(' '):
        if word in embeddings:
            res += embeddings[word]
            n += 1
    if n == 0:
        return np.zeros(dim)
    else:
        return res / float(n)


# To check the basic correctness of your implementation, run the function *question_to_vec_tests*.

# In[14]:


def question_to_vec_tests():
    if (np.zeros(300) != question_to_vec('', wv_embeddings)).any():
        return "You need to return zero vector for empty question."
    if (np.zeros(300) != question_to_vec('thereisnosuchword', wv_embeddings)).any():
        return "You need to return zero vector for the question, which consists only unknown words."
    if (wv_embeddings['word'] != question_to_vec('word', wv_embeddings)).any():
        return "You need to check the corectness of your function."
    if ((wv_embeddings['I'] + wv_embeddings['am']) / 2 != question_to_vec('I am', wv_embeddings)).any():
        return "Your function should calculate a mean of word vectors."
    if (wv_embeddings['word'] != question_to_vec('thereisnosuchword word', wv_embeddings)).any():
        return "You should not consider words which embeddings are unknown."
    return "Basic tests are passed."


# In[15]:


print(question_to_vec_tests())


# You can submit embeddings for the questions from the file *test_embeddings.tsv* to earn the points. In this task you don't need to transform the text of a question somehow.

# In[16]:


from util import array_to_string


# In[17]:


question2vec_result = []
for question in open('data/test_embeddings.tsv'):
    question = question.strip()
    answer = question_to_vec(question, wv_embeddings)
    question2vec_result = np.append(question2vec_result, answer)

grader.submit_tag('Question2Vec', array_to_string(question2vec_result))


# Now we have a method to create a representation of any sentence and we are ready for the first evaluation. So, let's check how well our solution (Google's vectors + *question_to_vec*) will work.
# 
# ## Evaluation of text similarity
# 
# We can imagine that if we use good embeddings, the cosine similarity between the duplicate sentences should be less than for the random ones. Overall, for each pair of duplicate sentences we can generate *R* random negative examples and find out the position of the correct duplicate.  
# 
# For example, we have the question *"Exceptions What really happens"* and we are sure that another question *"How does the catch keyword determine the type of exception that was thrown"* is a duplicate. But our model doesn't know it and tries to find out the best option also among questions like *"How Can I Make These Links Rotate in PHP"*, *"NSLog array description not memory address"* and *"PECL_HTTP not recognised php ubuntu"*. The goal of the model is to rank all these 4 questions (1 *positive* and *R* = 3 *negative*) in the way that the correct one is in the first place.
# 
# However, it is unnatural to count on that the best candidate will be always in the first place. So let us consider the place of the best candidate in the sorted list of candidates and formulate a metric based on it. We can fix some *K* — a reasonalble number of top-ranked elements and *N* — a number of queries (size of the sample).
# 
# ### Hits@K
# 
# The first simple metric will be a number of correct hits for some *K*:
# $$ \text{Hits@K} = \frac{1}{N}\sum_{i=1}^N \, [dup_i \in topK(q_i)]$$
# 
# where $q_i$ is the i-th query, $dup_i$ is its duplicate, $topK(q_i)$ is the top K elements of the ranked sentences provided by our model and the operation $[dup_i \in topK(q_i)]$ equals 1 if the condition is true and 0 otherwise (more details about this operation could be found [here](https://en.wikipedia.org/wiki/Iverson_bracket)).
# 
# 
# ### DCG@K
# The second one is a simplified [DCG metric](https://en.wikipedia.org/wiki/Discounted_cumulative_gain):
# 
# $$ \text{DCG@K} = \frac{1}{N} \sum_{i=1}^N\frac{1}{\log_2(1+rank_{dup_i})}\cdot[rank_{dup_i} \le K] $$
# 
# where $rank_{dup_i}$ is a position of the duplicate in the sorted list of the nearest sentences for the query $q_i$. According to this metric, the model gets a higher reward for a higher position of the correct answer. If the answer does not appear in topK at all, the reward is zero. 

# ### Evaluation examples
# 
# Let's calculate the described metrics for the toy example introduced above. In this case $N$ = 1 and the correct candidate for $q_1$ is *"How does the catch keyword determine the type of exception that was thrown"*. Consider the following ranking of the candidates:
# 1. *"How Can I Make These Links Rotate in PHP"*
# 2. *"How does the catch keyword determine the type of exception that was thrown"*
# 3. *"NSLog array description not memory address"*
# 4. *"PECL_HTTP not recognised php ubuntu"*
# 
# Using the ranking above, calculate *Hits@K* metric for *K = 1, 2, 4*: 
#  
# - [K = 1] $\text{Hits@1} = \frac{1}{1}\sum_{i=1}^1 \, [dup_i \in top1(q_i)] = [dup_1 \in top1(q_1)] = 0$ because the correct answer doesn't appear in the *top1* list.
# - [K = 2] $\text{Hits@2} = \frac{1}{1}\sum_{i=1}^1 \, [dup_i \in top2(q_i)] = [dup_1 \in top2(q_1)] = 1$ because $rank_{dup_1} = 2$.
# - [K = 4] $\text{Hits@4} = \frac{1}{1}\sum_{i=1}^1 \, [dup_i \in top4(q_i)] = [dup_1 \in top4(q_1)] = 1$
# 
# Using the ranking above, calculate *DCG@K* metric for *K = 1, 2, 4*:
# 
# - [K = 1] $\text{DCG@1} = \frac{1}{1} \sum_{i=1}^1\frac{1}{\log_2(1+rank_{dup_i})}\cdot[rank_{dup_i} \le 1] = \frac{1}{\log_2(1+rank_{dup_i})}\cdot[rank_{dup_i} \le 1] = 0$ because the correct answer doesn't appear in the top1 list.
# - [K = 2] $\text{DCG@2} = \frac{1}{1} \sum_{i=1}^1\frac{1}{\log_2(1+rank_{dup_i})}\cdot[rank_{dup_i} \le 2] = \frac{1}{\log_2{3}}$, because $rank_{dup_1} = 2$.
# - [K = 4] $\text{DCG@4} = \frac{1}{1} \sum_{i=1}^1\frac{1}{\log_2(1+rank_{dup_i})}\cdot[rank_{dup_i} \le 4] = \frac{1}{\log_2{3}}$.
# 

# **Tasks 2 and 3 (HitsCount and DCGScore).** Implement the functions *hits_count* and *dcg_score* as described above. Each function has two arguments: *dup_ranks* and *k*. *dup_ranks* is a list which contains *values of ranks* of duplicates. For example, *dup_ranks* is *[2]* for the example provided above.

# In[18]:


def hits_count(dup_ranks, k):
    """
        dup_ranks: list with ranks for each duplicate (the best rank is 1, the worst is len(dup_ranks))
        k: number of top-ranked elements

        result: float number
    """
    res = 0
    n = len(dup_ranks)
    for rank in dup_ranks:
        if rank <= k:
            res += 1
    return float(res) / float(n)


# Test your code on the tiny examples:

# In[19]:


def test_hits():
    # *Evaluation example*
    # answers — dup_i
    answers = ["How does the catch keyword determine the type of exception that was thrown"]
    
    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    
    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1, 1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function."
    
    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown", 
               "Convert Google results object (pure js) to Python object"]
    
    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: both duplicates on the first position in ranked list)."
        
    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: one candidate on the first position, another — on the second)."

    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: both candidates on the second position)."

    return "Basic test are passed."


# In[20]:


print(test_hits())


# In[21]:


def dcg_score(dup_ranks, k):
    """
        dup_ranks: list with ranks for each duplicate (the best rank is 1, the worst is len(dup_ranks))
        k: number of top-ranked elements

        result: float number
    """
    n = len(dup_ranks)
    res = 0.
    for rank in dup_ranks:
        if rank <= k:
            res += 1 / np.log2(1 + rank)
    return float(res) / float(n)


# In[22]:


def test_dcg():
    # *Evaluation example*
    # answers — dup_i
    answers = ["How does the catch keyword determine the type of exception that was thrown"]
    
    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    
    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1 / (np.log2(3)), 1 / (np.log2(3)), 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function."
    
    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown", 
               "Convert Google results object (pure js) to Python object"]

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: both duplicates on the first position in ranked list)."
        
    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, (1 + (1 / (np.log2(3)))) / 2]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: one candidate on the first position, another — on the second)."
        
    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: both candidates on the second position)."

    return "Basic test are passed."


# In[23]:


print(test_dcg())


# Submit results of the functions *hits_count* and *dcg_score* for the following examples to earn the points.

# In[24]:


test_examples = [
    [1],
    [1, 2],
    [2, 1],
    [1, 2, 3],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [9, 5, 4, 2, 8, 10, 7, 6, 1, 3],
    [4, 3, 5, 1, 9, 10, 7, 8, 2, 6],
    [5, 1, 7, 6, 2, 3, 8, 9, 10, 4],
    [6, 3, 1, 4, 7, 2, 9, 8, 10, 5],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
]


# In[25]:


hits_results = []
for example in test_examples:
    for k in range(len(example)):
        hits_results.append(hits_count(example, k + 1))
grader.submit_tag('HitsCount', array_to_string(hits_results))


# In[26]:


dcg_results = []
for example in test_examples:
    for k in range(len(example)):
        dcg_results.append(dcg_score(example, k + 1))
grader.submit_tag('DCGScore', array_to_string(dcg_results))


# ##  First solution: pre-trained embeddings

# We will work with predefined train, validation and test corpora. All the files are tab-separated, but have a different format:
#  - *train* corpus contains similar sentences at the same row.
#  - *validation* corpus contains the following columns: *question*, *similar question*, *negative example 1*, *negative example 2*, ... 
#  - *test* corpus contains the following columns: *question*, *example 1*, *example 2*, ...
# 
# Validation corpus will be used for the intermediate validation of models. The test data will be necessary for submitting the quality of your model in the system.

# Now you should upload *validation* corpus to evaluate current solution.

# In[27]:


def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split('\t'))
    return data


# In[28]:


validation = read_corpus('data/validation.tsv')


# In[29]:


from sklearn.metrics.pairwise import cosine_similarity


# We will use cosine distance to rank candidate questions which you need to implement in the function *rank_candidates*. The function should return a sorted list of pairs *(initial position in candidates list, candidate)*. Index of some pair corresponds to its rank (the first is the best). For example, if the list of candidates was *[a, b, c]* and the most similar is *c*, then *a* and *b*, the function should return a list *[(2, c), (0, a), (1, b)]*.
# 
# Pay attention, if you use the function *cosine_similarity* from *sklearn.metrics.pairwise* to calculate similarity because it works in a different way: most similar objects has greatest similarity. It's preferable to use a vectorized version of *cosine_similarity* function. Try to compute similarity at once and not use list comprehension. It should speed up your computations significantly.

# In[34]:


def rank_candidates(question, candidates, embeddings, dim=300):
    """
        question: a string
        candidates: a list of strings (candidates) which we want to rank
        embeddings: some embeddings
        dim: dimension of the current embeddings
        
        result: a list of pairs (initial position in the list, question)
    """
    mean_question = question_to_vec(question, embeddings, dim)
    mean_embeddings = [question_to_vec(candidate, embeddings, dim) for candidate in candidates]
    similarities = cosine_similarity([mean_question], mean_embeddings)
#     print(similarities)
    result = sorted(list(enumerate(candidates)), key=lambda x: -similarities[0,x[0]])
#     print(result)
    return result


# Test your code on the tiny examples:

# In[35]:


def test_rank_candidates():
    questions = ['converting string to list', 'Sending array via Ajax fails']
    candidates = [['Convert Google results object (pure js) to Python object', 
                   'C# create cookie from string and send it',
                   'How to use jQuery AJAX for an outside domain?'], 
                  ['Getting all list items of an unordered list in PHP', 
                   'WPF- How to update the changes in list item of a list', 
                   'select2 not displaying search results']]
    results = [[(1, 'C# create cookie from string and send it'), 
                (0, 'Convert Google results object (pure js) to Python object'), 
                (2, 'How to use jQuery AJAX for an outside domain?')],
               [(0, 'Getting all list items of an unordered list in PHP'), 
                (2, 'select2 not displaying search results'), 
                (1, 'WPF- How to update the changes in list item of a list')]]
    for question, q_candidates, result in zip(questions, candidates, results):
        ranks = rank_candidates(question, q_candidates, wv_embeddings, 300)
        if not np.all(ranks == result):
            return "Check the function."
    return "Basic tests are passed."


# In[36]:


print(test_rank_candidates())


# Now we can test the quality of the current approach. Run the next two cells to get the results. Pay attention that calculation of similarity between vectors takes time and this calculation is computed approximately in 10 minutes.

# In[37]:


wv_ranking = []
for line in validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, wv_embeddings)
    wv_ranking.append([r[0] for r in ranks].index(0) + 1)


# In[38]:


for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_ranking, k), k, hits_count(wv_ranking, k)))


# If you did all the steps correctly, you should be frustrated by the received results. Let's try to understand why the quality is so low. First of all, when you work with some data it is necessary to have an idea how the data looks like. Print several questions from the data:

# In[39]:


for line in validation[:3]:
    q, *examples = line
    print(q, *examples[:3])


# As you can see, we deal with the raw data. It means that we have many punctuation marks, special characters and unlowercased letters. In our case, it could lead to the situation where we can't find some embeddings, e.g. for the word "grid?". 
# 
# To solve this problem you should use the functions *text_prepare* from the previous assignments to prepare the data.

# In[40]:


from util import text_prepare


# Now transform all the questions from the validation set:

# In[41]:


prepared_validation = []
for line in validation:
    ######### YOUR CODE HERE #############
    prepared_validation.append([text_prepare(i) for i in line])


# Let's evaluate the approach again after the preparation:

# In[42]:


wv_prepared_ranking = []
for line in prepared_validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, wv_embeddings)
    wv_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)


# In[43]:


for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_prepared_ranking, k), 
                                              k, hits_count(wv_prepared_ranking, k)))


# Now, prepare also train and test data, because you will need it in the future:

# In[44]:


def prepare_file(in_, out_):
    out = open(out_, 'w')
    for line in open(in_, encoding='utf8'):
        line = line.strip().split('\t')
        new_line = [text_prepare(q) for q in line]
        print(*new_line, sep='\t', file=out)
    out.close()


# In[45]:


######################################
######### YOUR CODE HERE #############
######################################
prepare_file("data/train.tsv", "data/train_prepared.tsv")
prepare_file("data/test.tsv", "data/test_prepared.tsv")


# **Task 4 (W2VTokenizedRanks).** For each question from prepared *test.tsv* submit the ranks of the candidates to earn the points. The calculations should take about 3-5 minutes. Pay attention that the function *rank_candidates* returns a ranking, while in this case you should find a position in this ranking. Ranks should start with 1.

# In[46]:


from util import matrix_to_string


# In[47]:


w2v_ranks_results = []
prepared_test_data = "data/test_prepared.tsv" ######### YOUR CODE HERE #############
for line in open(prepared_test_data):
    q, *ex = line.strip().split('\t')
    ranks = rank_candidates(q, ex, wv_embeddings, 300)
    ranked_candidates = [r[0] for r in ranks]
    w2v_ranks_results.append([ranked_candidates.index(i) + 1 for i in range(len(ranked_candidates))])
    
grader.submit_tag('W2VTokenizedRanks', matrix_to_string(w2v_ranks_results))


# ## Advanced solution: StarSpace embeddings
# 
# Now you are ready to train your own word embeddings! In particular, you need to train embeddings specially for our task of duplicates detection. Unfortunately, StarSpace cannot be run on Windows and we recommend to use provided
# [docker container](https://github.com/hse-aml/natural-language-processing/blob/master/Docker-tutorial.md) or other alternatives. Don't delete results of this task because you will need it in the final project.
# 
# ### How it works and what's the main difference with word2vec?
# The main point in this section is that StarSpace can be trained specifically for some tasks. In contrast to word2vec model, which tries to train similar embeddings for words in similar contexts, StarSpace uses embeddings for the whole sentence (just as a sum of embeddings of words and phrases). Despite the fact that in both cases we get word embeddings as a result of the training, StarSpace embeddings are trained using some supervised data, e.g. a set of similar sentence pairs, and thus they can better suit the task.
# 
# In our case, StarSpace should use two types of sentence pairs for training: "positive" and "negative". "Positive" examples are extracted from the train sample (duplicates, high similarity) and the "negative" examples are generated randomly (low similarity assumed). 
# 
# ### How to choose the best params for the model?
# Normally, you would start with some default choice and then run extensive experiments to compare different strategies. However, we have some recommendations ready for you to save your time:
# - Be careful with choosing the suitable training mode. In this task we want to explore texts similarity which corresponds to *trainMode = 3*.
# - Use adagrad optimization (parameter *adagrad = true*).
# - Set the length of phrase equal to 1 (parameter *ngrams*), because we need embeddings only for words.
# - Don't use a large number of *epochs* (we think that 5 should be enough).
# - Try dimension *dim* equal to 100.
# - To compare embeddings usually *cosine* *similarity* is used.
# - Set *minCount* greater than 1 (for example, 2) if you don't want to get embeddings for extremely rare words.
# - Parameter *verbose = true* could show you the progress of the training process.
# - Set parameter *fileFormat* equals *labelDoc*.
# - Parameter *negSearchLimit* is responsible for a number of negative examples which is used during the training. We think that 10 will be enought for this task.
# - To increase a speed of training we recommend to set *learning rate* to 0.05.

# Train StarSpace embeddings for unigrams on the train dataset. You don't need to change the format of the input data. Just don't forget to use prepared version of the training data. 
# 
# If you follow the instruction, the training process will take about 1 hour.

# In[42]:


######### TRAINING HAPPENING HERE #############


# In[51]:


#!starspace train -trainFile "data/train_prepared.tsv" -model starspace_embeddings -adagrad true -ngrams 1 -lr 0.05 \
#-epoch 5 -dim 100 -negSearchLimit 10 -trainMode 3 -similarity "cosine" -minCount 2 -fileFormat labelDoc \
#-verbose true


# And now we can compare the new embeddings with the previous ones. You can find trained word vectors in the file *[model_file_name].tsv*. Upload the embeddings from StarSpace into a dict. 

# In[48]:


def read_starspace_embeddings(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split('\t')
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
    return word_to_vec_map


# In[49]:


starspace_embeddings = read_starspace_embeddings('starspace_embeddings.tsv')


# In[51]:


ss_prepared_ranking = []
for line in prepared_validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, starspace_embeddings, 100)
    ss_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)


# In[52]:


for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(ss_prepared_ranking, k), 
                                               k, hits_count(ss_prepared_ranking, k)))


# Due to training for the particular task with the supervised data, you should expect to obtain a higher quality than for the previous approach. In additiion, despite the fact that StarSpace's trained vectors have a smaller dimension than word2vec's, it provides better results in this task.

# **Task 5 (StarSpaceRanks).** For each question from prepared *test.tsv* submit the ranks of the candidates for trained representation.

# In[53]:


starspace_ranks_results = []
prepared_test_data = "data/test_prepared.tsv" ######### YOUR CODE HERE #############
for line in open(prepared_test_data):
    q, *ex = line.strip().split('\t')
    ranks = rank_candidates(q, ex, starspace_embeddings, 100)
    ranked_candidates = [r[0] for r in ranks]
    starspace_ranks_results.append([ranked_candidates.index(i) + 1 for i in range(len(ranked_candidates))])
    
grader.submit_tag('StarSpaceRanks', matrix_to_string(starspace_ranks_results))


# ### Authorization & Submission
# To submit assignment parts to Cousera platform, please, enter your e-mail and token into variables below. You can generate token on this programming assignment page. <b>Note:</b> Token expires 30 minutes after generation.

# In[54]:


STUDENT_EMAIL = "bhaskarjitsarmah@gmail.com" # EMAIL 
STUDENT_TOKEN = "ZWzvxUvwsUm3lmJB" # TOKEN 
grader.status()


# If you want to submit these answers, run cell below

# In[55]:


grader.submit(STUDENT_EMAIL, STUDENT_TOKEN)

