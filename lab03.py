# Cost 함수에 대한 시각화

import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# X and Y 데이터
X = [1, 2, 3]
Y = [1, 2, 3]
# x=1 일때 y=1이 되고 ...

# Variable -> Tensorflow가 임의로 바꿀 수 있는 값
# rank는 1
W = tf.placeholder(tf.float32)
hypothesis = X * W

# cost function
# reduce_mean -> 평균내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# show the cost function
plt.plot(W_val, cost_val)
plt.show()

print("------------------")


