import tensorflow as tf
import csv
import numpy as np
from tensorflow.python.platform import gfile
import time
import math

AIC_PATH = r"D:/project/data/20170916/ai_challenger_stock_train_20170916/"
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


def one_hot_matrix(labels, C):

    with tf.Session() as sess:
        C = tf.constant(C, name="C")
        one_hot_matrix = tf.one_hot(labels, C, axis=1)
        one_hot = sess.run(one_hot_matrix)

    return one_hot

def main():
    # data_save()
    data = np.load("data.npy")
    target = np.load("target.npy")

    hidden_size = 10
    learning_rate = 0.01
    class_num = 2
    num_epochs = 2500
    train_size = 4000
    test_begin = train_size+6000
    test_size = 400

    train_data = data[0:train_size, 1:-3]
    test_data = data[test_begin:test_begin+test_size, 1:-3]
    train_target = target[0:train_size]
    test_target = target[test_begin:test_begin+test_size]
    train_labels = one_hot_matrix(train_target, C=class_num)
    test_labels = one_hot_matrix(test_target, C=class_num)
    print(train_data.shape, train_labels.shape)
    X = tf.placeholder(tf.float32, shape=(None, train_data.shape[1]))
    Y = tf.placeholder(tf.int32, shape=(None, class_num))

    input_layer = X
    input_size = train_data.shape[1]
    output_size = train_data.shape[1]
    for i in range(hidden_size):
        input_layer = add_layer(input_layer, input_size, output_size, activation_function=tf.nn.relu)

    logits = add_layer(input_layer, input_size, class_num, activation_function=None)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='xentropy')
    cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            start_time = time.time()
            _, loss_value = sess.run([optimizer, cost], feed_dict={X: train_data, Y: train_labels})
            duration = time.time() - start_time

            # Print the cost every epoch
            if epoch % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (epoch, loss_value, duration))
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print ("Train Accuracy:", accuracy.eval({X: train_data, Y: train_labels}))
                loss_value = sess.run(cost, feed_dict={X: test_data, Y: test_labels})
                print("test loss:%.2f Test Accuracy:"%(loss_value), accuracy.eval({X: test_data, Y: test_labels}))

main()
