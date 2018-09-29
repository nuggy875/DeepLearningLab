import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# constant
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(2.0)
node3 = tf.add(node1, node2)

# placeholder 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

# 실행
sess = tf.Session()

print(sess.run([node1, node2]))
print(sess.run(node3))

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

print("------------------")


