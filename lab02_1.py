# Linear Regression (Place holder 를 이용하여
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# X and Y 데이터를 placeholder로 정의 (나중에 넣어 줄거야)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Variable -> Tensorflow가 임의로 바꿀 수 있는 값
# rank는 1
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = X * W + b

# cost function
# reduce_mean -> 평균내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize (cost function 값을 최소화 하는 경사하강법)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [3, 5, 7, 9, 11]})

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print("------------------")

# Train 된 model 에 대한 Testing

print(sess.run(hypothesis, feed_dict={X: [7]}))
print(sess.run(hypothesis, feed_dict={X: [2.4]}))
print(sess.run(hypothesis, feed_dict={X: [3.1, 9.9]}))


