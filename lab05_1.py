# Logistic Classification 구현하기 (당뇨병 예측하기)

import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# X -> [공부시간, 논 시간] 이라고 해보자
# Y -> 0 fail, 1 pass
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))


hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    hyp_val, cost_val, _ = sess.run([hypothesis, cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, "::", cost_val)


# 0.5보다 작으면 False, 0.5보다 크면 True 라고 둔다.

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
print("\n Hypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

