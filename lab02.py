# Linear Regression
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# X and Y 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]
# x=1 일때 y=1이 되고 ...

# Variable -> Tensorflow가 임의로 바꿀 수 있는 값
# rank는 1
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x_train * W + b

# cost function
# reduce_mean -> 평균내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize (cost function 값을 최소화 하는 경사하강법)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, "cost:", sess.run(cost), "W:", sess.run(W), "b:", sess.run(b))


print("------------------")


