# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:30:37 2018

@author: gary.roberts
"""

import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def print_section(text):
    print("\n")
    print("--------------------------------")
    print(str(text))
    print("--------------------------------")
    
# download data
print_section("Downloading data...")

# Note that pandas.read_csv will read anything with a read() method, so technically
# I could have passed the response to it.
#response = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
#irisdata = response.read()

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=["sepal length", "sepal width", "petal length", "petal width", "class"])

print(df.head())

# Check data has downloaded
print_section("Initial Testing...")
assert df.shape == (150, 5)
assert df.loc[0,"sepal length"] == 5.1
print("All tests passed.")

# Shuffle data
print_section("Shuffling...")
df=shuffle(df).reset_index(drop=True)
print(df.head())

# Visualising
print_section("Generating plot of data...")
g=sns.pairplot(df, hue="class", size= 2.5)
#plt.figure(figsize=(15,15))
#sns.heatmap(df.corr(),annot = True,fmt = ".2f",cbar = True)
#plt.xticks(rotation = 90)
#plt.yticks(rotation = 0)

print("Done")

# Condition data to have zero mean and equal variance (normalise) and measure.
print_section("Normalising...")
for feature in df.loc[:,"sepal length":"petal width"]:
    df[feature] = (df[feature] - df[feature].mean())/df[feature].std()
print("Done")

print(df.head())

# Check condition of data
print_section("Checking data...")
print("Averages")
print(df.mean())
print("Deviations")
print(pow(df.std(),2))

# New create training and test set
y=pd.get_dummies(df.loc[:,"class"])
X=df.loc[:,"sepal length":"petal width"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Check size of training and test data
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

########################################
# TensorFlow section below
########################################

# tunable parameters
learning_rate = 0.5
num_steps = 1050

print_section("Setup Tensorflow Variables")

# placeholder for Iris data
x = tf.placeholder(tf.float32, shape=[None,4])

# placeholder for vector containing probabilities of it being each class (setosa, )
y_ = tf.placeholder(tf.float32, shape=[None,3])

W = tf.Variable(tf.zeros([4,3]))
b = tf.Variable(tf.zeros([3]))

#inference model - softmax regression on a matrix multiplication
y = tf.nn.softmax(tf.matmul(x,W) +b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# each training step in gradient decent we want to minimize cross entropy
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# initialize the global variables
init = tf.global_variables_initializer()

print("Done...")

print_section("Run TensorFlow Training")
# create an interactive session that can span multiple code blocks.  Don't 
# forget to explicity close the session with sess.close()
sess = tf.Session()

# perform the initialization which is only the initialization of all global variables
sess.run(init)

# Perform 1000 training steps
for i in range (num_steps):
    sess.run(training_step, feed_dict={x: X_train, y_: y_train})
print("Done...")
print_section("Run TensorFlow Testing")
# Evaluate how well the model did. Do this by comparing the digit with the highest probability in 
#    actual (y) and predicted (y_).
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
print("Test Accuracy: {0:.2f}%".format(test_accuracy * 100.0))

#final_b_values = sess.run(b)
#final_W_values = sess.run(W)

#print(final_b_values)
#print(final_W_values)

sess.close()

