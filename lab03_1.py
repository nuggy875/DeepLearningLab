# Cost 함수에 대한 시각화

import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# X and Y 데이터
x_data = [1, 2, 3]
y_data = [1, 2, 3]
# x=1 일때 y=1이 되고 ...

# Variable -> TensorFlow가 임의로 바꿀 수 있는 값
# rank는 1
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

# cost function
cost = tf.reduce_sum(tf.square(hypothesis - Y))


# Cost function 을 Minimize 하는 function을 직접 만들었다.
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

#   Minimize (cost function 값을 최소화 하는 경사하강법)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# 다 미분할 필요 없이 optimizer 를 사용하면 간단히 할 수 있다!

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print("------------------")