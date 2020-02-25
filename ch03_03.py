import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(100.0)

hypothesis = W * X

gradient = tf.reduce_mean((W * X - Y) * X) * 2

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost)
ag = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    print(i, sess.run([gradient, W, gvs]))
    sess.run(ag)
