#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x1_data = [73.0, 93.0, 89.0, 96.0, 73.0]
x2_data = [80.0, 88.0, 91.0, 98.0, 66.0]
x3_data = [75.0, 93.0, 90.0, 100.0, 70.0]
Y_data = [152.0, 185.0, 180.0, 196.0, 142.0]

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(3001):
    cost_val, w1v, w2v, w3v, bv, _ = sess.run([cost, w1, w2, w3, b, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: Y_data})
    if step % 30 == 0:
        print(step, cost_val, w1v, w2v, w3v, bv)
