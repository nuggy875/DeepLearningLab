import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

xy_data = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = xy_data[:, 0:-1]
y_data = xy_data[:, [-1]]

print(x_data.shape)
print(x_data)
print(y_data.shape)
print(y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    cost_val, hyp_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if i % 20 == 0:
        print('cost:', cost_val, '  hyp_val:', hyp_val)

print(sess.run(hypothesis, feed_dict={X: [[10, 20, 30], [20, 30, 140]]}))
