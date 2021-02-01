import tensorflow as tf

node1 = tf.constant(5)
customNode = tf.placeholder(tf.int32, [], name="custom")

op = tf.add(node1, customNode)

sess = tf.Session()

result = sess.run(op, feed_dict={customNode: 100})
print(result)
