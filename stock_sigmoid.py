import tensorflow as tf
import csv
import numpy as np
from tensorflow.python.platform import gfile
import math

AIC_PATH = r"D:/project/AIchallenger/data/20170916/ai_challenger_stock_train_20170916/"
AIC_TRAINING = AIC_PATH + "stock_train_data_20170916.csv"

print(AIC_TRAINING)


def stack_csv_load(filename,
                   target_dtype,
                   features_dtype,
                   target_column=-1):
    """Load dataset from CSV file without a header row."""
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data, target = [], []
        counts = 0
        for row in data_file:
            counts += 1
            if counts == 1:
                continue
            # if counts == 100:
            #   break
            target.append(row.pop(target_column))
            data.append(np.asarray(row, dtype=features_dtype))

    target = np.array(target, dtype=np.float32)
    target = np.array(target, dtype=target_dtype)
    data = np.array(data)
    return {"data":data, "target":target}


def data_save():
    training_set = stack_csv_load(filename=AIC_TRAINING,
                                  target_dtype=np.int32,
                                  features_dtype=np.float32,
                                  target_column=-3)
    np.save("data", training_set["data"])
    np.save("target", training_set["target"])


def add_layer(inputs,
              in_size,
              out_size,
              activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size],
                                           stddev=1.0 / math.sqrt(float(in_size))))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def dense(x,
          size,
          phase,
          keep_prob,
          scope):
    with tf.variable_scope(scope):
        up = tf.contrib.layers.fully_connected(x, size,
                                               activation_fn=None,
                                               scope='dense')

        up = tf.contrib.layers.batch_norm(up,
                                      center=True, scale=True,
                                      is_training=phase,
                                      scope='bn')

        up = tf.nn.dropout(up, keep_prob, name="do")

    return up


def dense_relu(x, size, keep_prob, scope):
    with tf.variable_scope(scope):
        h1 = dense(x, size, 'dense')
        h1_drop = tf.nn.dropout(h1, keep_prob)
        return tf.nn.relu(h1_drop, 'relu')


def dense_batch_relu(x, size, phase, keep_prob, scope):
    with tf.variable_scope(scope):
        up = tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope='dense')

        up = tf.contrib.layers.batch_norm(up,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')

        up = tf.nn.dropout(up, keep_prob, name="do")

        return tf.nn.relu(up, 'relu')


def one_hot_matrix(labels,
                   C):
    with tf.Session() as sess:
        C = tf.constant(C, name="C")
        one_hot_matrix = tf.one_hot(labels, C, axis=1)
        one_hot = sess.run(one_hot_matrix)

    return one_hot


def data_get_whatever(data, target):

    train_size = 200000
    test_begin = 250000+16000
    test_size = 20000
    class_num = 2

    train_data = data[0:train_size, 1:-3]
    test_data = data[test_begin:test_begin+test_size, 1:-3]
    train_target = target[0:train_size]
    test_target = target[test_begin:test_begin+test_size]
    train_labels = one_hot_matrix(train_target, C=class_num)
    test_labels = one_hot_matrix(test_target, C=class_num)
    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)

    data_set = {"train_data":train_data,
           "train_labels": train_labels,
           "test_data": test_data,
           "test_labels": test_labels}
    return data_set


def data_get_group(data, target, group, split):

    class_num = 2

    target = one_hot_matrix(target, C=class_num)

    group_index = np.where(data[:, 90] == group)
    group_data = data[group_index]
    group_lables = target[group_index]

    group_train_index = np.where(group_data[:, 91] < split)
    group_test_index = np.where(group_data[:, 91] >= split)

    train_data =group_data[group_train_index]
    train_data = train_data[:, 1:89]
    train_labels = group_lables[group_train_index]

    test_data = group_data[group_test_index]
    test_data = test_data[:, 1:89]
    test_labels = group_lables[group_test_index]

    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)

    data_set = {"train_data":train_data,
           "train_labels": train_labels,
           "test_data": test_data,
           "test_labels": test_labels}
    return data_set


def random_mini_batches(X, Y, mini_batch_size=64, seed=None):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: (k + 1) * mini_batch_size, :]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:, :]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def main():
    # data_save()
    data = np.load("data.npy")
    target = np.load("target.npy")

    hidden_size = 2
    learning_rate = 0.1
    dropout_keep = 0.2

    num_epochs = 2500000

    # data_set = data_get_whatever(data, target)
    data_set = data_get_group(data, target, 2, 15)

    train_data = data_set["train_data"]
    train_labels = data_set["train_labels"]
    test_data = data_set["test_data"]
    test_labels = data_set["test_labels"]

    X = tf.placeholder(tf.float32, shape=(None, train_data.shape[1]), name="X")
    Y = tf.placeholder(tf.int32, shape=(None, train_labels.shape[1]), name='Y')
    phase = tf.placeholder(tf.bool, name='phase')
    keep_prob = tf.placeholder(tf.float32)

    input_layer = X

    output_size = int(train_data.shape[1])
    print(output_size)
    for i in range(hidden_size):
        input_layer = dense_batch_relu(input_layer, output_size, phase, keep_prob, "layer"+str(i+i))

    logits = dense(input_layer, train_labels.shape[1], phase, keep_prob, "layer_final")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='xentropy')
    cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.summary.scalar('loss', cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("logs/train_%d_%f_%f"%(hidden_size, learning_rate, dropout_keep), sess.graph)
        test_writer = tf.summary.FileWriter("logs/test_%d_%f_%f"%(hidden_size, learning_rate, dropout_keep), sess.graph)

        for epoch in range(num_epochs):
            minibatches = random_mini_batches(train_data, train_labels, 100)

            for minibatch in minibatches:
                _, loss_value = sess.run([optimizer, cost], feed_dict={X: minibatch[0], Y: minibatch[1],
                                                                       phase:False, keep_prob:dropout_keep})
                # Print the cost every epoch
            if epoch % 1 == 0:
                train_result, loss_train = sess.run([merged, cost], feed_dict={X: train_data, Y: train_labels, phase: False, keep_prob: 1})
                acc_train = accuracy.eval({X: train_data, Y: train_labels, phase:False, keep_prob:1})
                test_result, loss_test = sess.run([merged, cost], feed_dict={X: test_data, Y: test_labels, phase:False, keep_prob:1})
                acc_test = accuracy.eval({X: test_data, Y: test_labels, phase: False, keep_prob: 1})
                print("Test loss:%.8f Test Accuracy:%.8f  |  Train loss:%.8f Train Accuracy:%.8f  epoch:%d"
                      %(loss_test, acc_test, loss_train, acc_train, epoch))
                train_writer.add_summary(train_result, epoch)
                test_writer.add_summary(test_result, epoch)


main()
