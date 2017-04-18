import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

# Parameters
learning_rate = 0.001
training_epochs =1000
batch_size = 25
display_step = 1

# Network Parameters
n_hidden_1 = 256   # 1st layer number of features
n_hidden_2 = 128   # 2nd layer number of features
n_input = 16     # data input (img shape: 28*28)
n_classes = 26   # total classes (A-Z letters)

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])



# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Read Data
read_data = pd.read_csv("./letter_recognition_training_data_set.csv")
label_data = pd.get_dummies(read_data, sparse=True)


test_data = pd.read_csv("./letter_recognition_testing_data_set.csv")

# Traning Data
train_input = label_data.loc[:16999, label_data.columns[:16]]
train_output = label_data.loc[:16999, label_data.columns[-26:]]

validation_input = label_data.loc[17000:, label_data.columns[:16]]
validation_output = label_data.loc[17000:, label_data.columns[-26:]]


saver = tf.train.Saver()

model_dir = "model"
model_name = "letter"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Construct model
pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Cost Array
costlist = []
epochlist = []
# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_input.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: train_input.loc[i*batch_size:(i+1)*batch_size-1],
                                                          y: train_output.loc[i*batch_size:(i+1)*batch_size-1]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        costlist.append(avg_cost)
        epochlist.append(epoch+1)
    print ("Optimization Finished!", "total batch" )

    saver.save(sess, os.path.join(model_dir, model_name))

    plt.xlabel("EPOCH")
    plt.ylabel("COST")
    plt.plot(epochlist, costlist, color="red")
    plt.savefig('costfig.png')
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x:validation_input , y: validation_output}))






