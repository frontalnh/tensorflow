import tensorflow as tf

# Variable is used for tensorflow which means trainable variable
# 학습과정에서 tensorflow 가 바꾼다.


# tf.random_normal([1]) => 임의값을 할당함
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost)
# apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train, feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
