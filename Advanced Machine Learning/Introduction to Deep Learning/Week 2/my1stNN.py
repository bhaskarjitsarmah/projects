
# coding: utf-8

# ### Loading Dataset

# In[1]:

import tensorflow as tf


# In[2]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)


# ### Parameters

# In[33]:

# parameter

learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1

n_input = 784  # 28 * 28 = 784
n_classes = 10  


# ### hidden layers

# In[4]:

n_hidden_layer = 256 # number of neurons in hidden layer


# ### Weights and Biases

# In[15]:

weights = {'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
          'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))}

biases = {'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
         'out': tf.Variable(tf.random_normal([n_classes]))}


# ### Input

# In[16]:

x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])


# ### Hidden layer with activation

# In[17]:

layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.sigmoid(layer_1)


# ### Output layer

# In[18]:

logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])


# ### Loss and Optimizer

# In[19]:

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# ### Session

# In[20]:

init = tf.global_variables_initializer()


# In[34]:

from sklearn.metrics import roc_auc_score
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            cost_epoch = sess.run(cost, {x: batch_x, y: batch_y})
        if(epoch % 10 == 0):
            print("loss at epoch %d:%.4f" % (epoch, cost_epoch))

