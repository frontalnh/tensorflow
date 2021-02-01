import tensorflow as tf

node1 = tf.constant(5)
node2 = tf.constant(6)

node3 = tf.add(node1, node2)

sess = tf.Session()

result = sess.run(node3)

print(result)

strNode = tf.constant("Hello World!")
print(sess.run(strNode))
