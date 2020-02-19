import tensorflow as tf

#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)

@tf.function
def adder_node(x, y):
    return x + y

res = adder_node(3, 3)
print(res)
