import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis-Y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-3, 5):
    current_cost, current_W = sess.run([cost, W], feed_dict={W: i})
    W_val.append(current_W)
    cost_val.append(current_cost)

plt.plot(W_val, cost_val)
plt.show()
