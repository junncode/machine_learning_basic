#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

'''
@tf.function
def hypothesis(x):
    return W * x + b
'''
hypothesis = x_train * W + b
'''
@tf.function
def cost(hypothesis):
    return tf.reduce_mean(tf.sqaure(hypothesis - y_train))
'''
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

'''
@tf.function
def train():
    optimizer = tf.train.GradientDescentOptimizer(learing_rate=0.01)
    ret = optimizer.minimize(cost)
    return ret
'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
