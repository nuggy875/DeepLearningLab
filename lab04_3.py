# Lab 4 Multi-variable linear regression (file input)
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
# [:, 0:-1]
# 전체 데이터에 대해서 ,  처음부터 마지막 하나를 제외한 걸 가져온다
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
# 데이터가 맞게 들어갔는지 확인하고 가자
print('----- X의 데이터 -----')
print(x_data.shape)
print(len(x_data))
print(x_data)

print('----- Y의 데이터 -----')
print(y_data.shape)
print(len(y_data))
print(y_data)

# placeholders for a tensor that will be always fed.
# Nx3, Nx1
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
# Testing
print('----- Testing Score -----')
print("100점, 70점, 101점이면 Your score will be ", sess.run(
    hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
