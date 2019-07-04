import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = tf.add(a, b)  # a + b

sess = tf.Session()
sess.run(adder_node, feed_dict={a: 3, b: 4})
sess.run(adder_node, feed_dict={a: [4, 5], b: [1, 3, 5]})
