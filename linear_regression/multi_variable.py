import tensorflow as tf
import numpy as np

xy = np.loadtxt('./linear_regression/data/scores.csv',
                delimiter=',', dtype=np.float)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# x_data = [[73., 80., 75], [93., 88., 93.]]
# y_data = [[152.], [185.]]

# Read with file name queue START
filename_queue = tf.train.string_input_producer(
    ['./data/scores.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch(
    [xy[0:-1], xy[-1:]], batch_size=10)

coord = tf.train.Coordinator()

sess = tf.Session()

threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Read with file name queue END

X = tf.placeholder(tf.float32, shape=[None, 3])  # 3개의 속성을 가진 데이터목록
Y = tf.placeholder(tf.float32, shape=[None, 1])  # 1개의 속성을 가진 output

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W)+b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(cost)


sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})


coord.request_stop()
coord.join(threads)
