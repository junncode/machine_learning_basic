#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

filename_queue = tf.train.string_input_producer(['ch04_data.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

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

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_batch, Y: y_batch})
    if step % 100 == 0:
        print(step, "Cost: ", cost_val, "\nPredict: ", hy_val)

# 예측하기
print("[100, 70, 101] score will be ", sess.run(hypothesis, feed_dict={x: [[100, 70, 101]]}))
print("[60, 70, 110] score will be ", sess.run(hypothesis, feed_dict={x: [[60, 70, 110]]}))

coord.request_stop()
coord.join(threads)
