# -*- coding: utf-8 -*-
from numpy.random import seed
seed(13)
from tensorflow import set_random_seed
set_random_seed(1)
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import math

def outlier_hunt(df):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than 2 outliers.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in df.columns.tolist():
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 1)

        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 99)

        # Interquartile rrange (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v >= 2)

    return multiple_outliers

data = pd.read_csv("train.csv")

# delete some bad data
remove_temp = data.drop(['Category'], 1)
remove_index = outlier_hunt(remove_temp)
data = data.drop(remove_index, axis=0)
data = data.reset_index(drop=True)


data = data[data['Attribute6']<0]
data = data.reset_index(drop=True)


X_data = data.drop(['Category'], 1)
Y_data = data['Category']

test_data = pd.read_csv("test.csv")


X_data['Attribute6'] = X_data['Attribute6'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute6'] = test_data['Attribute6'].map(lambda x: x/abs(x)*np.log(abs(x)))
'''
X_data['Attribute6'] = X_data['Attribute6'].map(lambda x: np.log(abs(x)))
test_data['Attribute6'] = test_data['Attribute6'].map(lambda x: np.log(abs(x)))
'''

X_data['Attribute4'] = X_data['Attribute4'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute4'] = test_data['Attribute4'].map(lambda x: x/abs(x)*np.log(abs(x)))

X_data['Attribute1'] = X_data['Attribute1'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute1'] = test_data['Attribute1'].map(lambda x: x/abs(x)*np.log(abs(x)))

'''
X_data['Attribute5'] = X_data['Attribute5'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute5'] = test_data['Attribute5'].map(lambda x: x/abs(x)*np.log(abs(x)))

X_data['Attribute2'] = X_data['Attribute2'].map(lambda x: x/abs(x)*np.log(abs(x)))
test_data['Attribute2'] = test_data['Attribute2'].map(lambda x: x/abs(x)*np.log(abs(x)))

'''

T = X_data.append(test_data)
scaler = StandardScaler()
scaler.fit(T)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_data = scaler.transform(X_data)
test_data = scaler.transform(test_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0)
Y_train = np.asarray(Y_train)

# Parameters
learning_rate = 0.001
num_steps = 4000
batch_size = 500
display_step = 50

num_input = 6
num_classes = 6

layer_dimension = [num_input, 100, 120, 140, 160, 140, 120, num_classes]

# dropout
keep = 0.9

regularizer_rate = 0.001


def get_weight(shape, lambd,i):
    #var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    #var = tf.get_variable(name="weight"+str(i),shape=(shape),dtype=tf.float32,initializer=tf.glorot_normal_initializer(seed=None,dtype=tf.float32))
    var = tf.Variable(tf.truncated_normal((shape), stddev=math.sqrt(2.0/shape[0])), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))

    return var


# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("int64", [None, ])
keep_prob = tf.placeholder(tf.float32)

n_layers = len(layer_dimension)
cur_layer = X
in_dimension = layer_dimension[0]


for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], regularizer_rate,i)
    bias = tf.Variable(tf.constant(0.0, shape=[out_dimension]))

    cur_layer = tf.matmul(cur_layer, weight) + bias

    if i != n_layers - 1:
        # cur_layer = tf.nn.relu(cur_layer)
        cur_layer = tf.nn.relu(cur_layer)

        if i != n_layers - 2:
            cur_layer = tf.nn.dropout(cur_layer, keep_prob)
    in_dimension = layer_dimension[i]

# mse_loss = tf.reduce_mean(tf.square)

# Construct model
logits = cur_layer
prediction = tf.nn.softmax(logits)
prediction_result = tf.argmax(prediction, 1)

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#    logits=logits, labels=Y))
loss_m = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
loss_s = tf.reduce_mean(tf.square(prediction_result - Y))
tf.add_to_collection('losses', loss_m)

loss_op = tf.add_n(tf.get_collection('losses'))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    #tf.set_random_seed(1)
    for step in range(1, num_steps + 1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        for batch_x, batch_y in minibatches(X_train, Y_train, batch_size, shuffle=False):
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: keep})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_m, accuracy], feed_dict={X: X_train,
                                                                Y: Y_train, keep_prob: keep})
            #print("Step " + str(step) + ", Minibatch Loss= " +
            #      str(loss) + ", Training Accuracy= " +
            #      str(acc))

    #print("Optimization Finished!")

    pre = sess.run(prediction_result, feed_dict={X: test_data, keep_prob: 1.0})
    ts = pd.DataFrame(pre)
    
    ts = pd.DataFrame({"Id":range(1,len(ts)+1),"Category":pre})
    cols = list(ts)
    cols.insert(0, cols.pop(cols.index('Id')))
    ts = ts.ix[:, cols]
    ts.to_csv("11510785-submission.csv",index = False,header=True)