#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

xy = np.loadtxt('ch04_data.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
Y_data = xy[:, [-1]]
x = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x: x_data, Y: Y_data})
    if step % 1000 == 0:
        print(step, "Cost: ", cost_val, "\nPredict: ", hy_val)

# 예측하기
print("[100, 70, 101] score will be ", sess.run(hypothesis, feed_dict={x: [[100, 70, 101]]}))
print("[60, 70, 110] score will be ", sess.run(hypothesis, feed_dict={x: [[60, 70, 110]]}))
