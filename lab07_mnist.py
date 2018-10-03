import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Python GUI 이용하여
import matplotlib.pyplot as plt
import random

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# 데이터 읽어와라 (읽어올 떄 One hot 으로 처리)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 1부터 10까지의 숫
nb_classses = 10
# 전체 데이터 set을 한 번 학습시킨 것 -> 1 Epoch  -> 15번 똘꺼야
training_epochs = 15
# 한 번에 몇개 씩 학습?
batch_size = 100
# 1000 개의 Training example, batch size 500 -> 1 epoch 에는 batch 를 두 번 돌면 됨


# 손글씨 픽셀 28 x 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 10 개의 출력 (One - hot Encoding)
Y = tf.placeholder(tf.float32, [None, nb_classses])

W = tf.Variable(tf.random_normal([784, nb_classses]))
b = tf.Variable(tf.random_normal([nb_classses]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test Model (Y 값과 hypothesis 를 비교하여 정확도를 측정
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 일반적인 학습 방법 -> 기억하자!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 15번 돌꺼야 (15번 Epoch)
    for epoch in range(training_epochs):
        avg_cost = 0
        # 전체 data size를 batch_size로 나누면 -> 몇 번 batch 를 돌았을 때 one epoch 이 나올 것인가가 나옴.
        total_batch = int(mnist.train.num_examples / batch_size)

        # 따라서 total_batch 만큼 돈다. (for 전체 -> 1 epoch)
        for i in range(total_batch):
            # 100개의 X, Y training data를 불러온다.
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    print('----- Learing Finished -----')
    # Testing
    # 학습하지 않았던 training data 를 이용
    print("Accuracy ", accuracy.eval(session=sess,
                                     feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


    r = random.randint(0, mnist.test.num_examples - 1)
    # argmax 를 이용해서 Label이 몇 인가? -> 1인가 2인가 ... 6인가
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    # 예측하는 것
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1),
                                  feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(mnist.test.images[r:r + 1]
               .reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

