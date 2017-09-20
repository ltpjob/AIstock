import tensorflow as tf
import csv
import numpy as np
from tensorflow.python.platform import gfile
import time
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

        # up = tf.contrib.layers.batch_norm(up,
        #                               center=True, scale=True,
        #                               is_training=phase,
        #                               scope='bn')

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

        # up = tf.contrib.layers.batch_norm(up,
        #                                   center=True, scale=True,
        #                                   is_training=phase,
        #                                   scope='bn')

        up = tf.nn.dropout(up, keep_prob, name="do")

        return tf.nn.tanh(up, 'tanh')


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


def main():
    # data_save()
    data = np.load("data.npy")
    target = np.load("target.npy")

    hidden_size = 5
    learning_rate = 0.1
    dropout_keep = 0.7
    class_num = 2

    num_epochs = 2500000
    # train_size = 200000
    # test_begin = 250000+16000
    # test_size = 20000
    #
    # train_data = data[0:train_size, 1:-3]
    # test_data = data[test_begin:test_begin+test_size, 1:-3]
    # train_target = target[0:train_size]
    # test_target = target[test_begin:test_begin+test_size]
    # train_labels = one_hot_matrix(train_target, C=class_num)
    # test_labels = one_hot_matrix(test_target, C=class_num)
    # print(train_data.shape, train_labels.shape)

    # data_set = data_get_whatever(data, target)
    data_set = data_get_group(data, target, 1, 19)

    train_data = data_set["train_data"]
    train_labels = data_set["train_labels"]
    test_data = data_set["test_data"]
    test_labels = data_set["test_labels"]

    X = tf.placeholder(tf.float32, shape=(None, train_data.shape[1]), name="X")
    Y = tf.placeholder(tf.int32, shape=(None, class_num), name='Y')
    phase = tf.placeholder(tf.bool, name='phase')
    keep_prob = tf.placeholder(tf.float32)

    input_layer = X

    # input_size = train_data.shape[1]
    # output_size = train_data.shape[1]
    # for i in range(hidden_size):
    #     input_layer = add_layer(input_layer, input_size, output_size, activation_function=tf.nn.relu)
    #
    # logits = add_layer(input_layer, input_size, class_num, activation_function=None)

    output_size = int(train_data.shape[1]*3)
    print(output_size)
    for i in range(hidden_size):
        input_layer = dense_batch_relu(input_layer, output_size, phase, keep_prob, "layer"+str(i+i))

    logits = dense(input_layer, class_num, phase, keep_prob, "layer_final")

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
            start_time = time.time()
            _, loss_value = sess.run([optimizer, cost], feed_dict={X: train_data, Y: train_labels,
                                                                   phase:False, keep_prob:dropout_keep})
            duration = time.time() - start_time
            # print('Step %d: loss = %.8f (%.3f sec)' % (epoch, loss_value, duration))

            # Print the cost every epoch
            if epoch % 100 == 0:
                # print('Step %d: loss = %.8f (%.3f sec)' % (epoch, loss_value, duration))

                # print ("train loss:%.8f Train Accuracy:"%(loss_value), accuracy.eval({X: train_data, Y: train_labels,
                #                                                                       phase:True, keep_prob:1}))

                train_result, loss_train = sess.run([merged, cost], feed_dict={X: train_data, Y: train_labels, phase: False, keep_prob: 1})
                acc_train = accuracy.eval({X: train_data, Y: train_labels, phase:False, keep_prob:1})
                test_result, loss_test = sess.run([merged, cost], feed_dict={X: test_data, Y: test_labels, phase:False, keep_prob:1})
                acc_test = accuracy.eval({X: test_data, Y: test_labels, phase: False, keep_prob: 1})
                print("Test loss:%.8f Test Accuracy:%.8f  |  Train loss:%.8f Train Accuracy:%.8f  epoch:%d"
                      %(loss_test, acc_test, loss_train, acc_train, epoch))
                train_writer.add_summary(train_result, epoch)
                test_writer.add_summary(test_result, epoch)


main()
