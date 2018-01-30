import tensorflow as tf


if __name__ == "__main__":

#    x_train = [1,2,3]
#    y_train = [1,2,3]

    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

#    hypothesis = x_train * W + b
    hypothesis = X * W + b

#    cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()

    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
#        sess.run(train)
#        if step % 20 == 0:
#            print(step, sess.run(cost), sess.run(W), sess.run(b))
        cost_val, W_val, b_val, _ = \
                sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)



