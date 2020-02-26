#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[73.0, 80., 75.], [93.0, 88., 93.], [89.0, 91., 90.], [96.0, 98., 100.], [73.0, 66., 70.]]
Y_data = [[152.0], [185.0], [180.0], [196.0], [142.0]]

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

x = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(3001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x: x_data, Y: Y_data})
    if step % 100 == 0:
        print(step, "Cost: ", cost_val, "\nPredict: ", hy_val)
