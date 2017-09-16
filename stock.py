import tensorflow as tf
import csv
import numpy as np
from tensorflow.python.platform import gfile

AIC_PATH = r"D:/project/AIchallenger/data/20170910/ai_challenger_stock_train_20170910/"
AIC_TRAINING = AIC_PATH + "stock_train_data_20170910.csv"

print(AIC_TRAINING)

def stack_csv_load(filename, target_dtype, features_dtype, target_column=-1):
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
    training_set = stack_csv_load(filename=AIC_TRAINING, target_dtype=np.int32, features_dtype=np.float32,
                                  target_column=-3)
    np.save("data", training_set["data"])
    np.save("target", training_set["target"])

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def main():
    # data_save()
    data = np.load("data.npy")
    target = np.load("target.npy")

    data = data[:, 1:]
    labels = target
    print(data.shape, target.shape)
    data_placeholder = tf.placeholder(tf.float32, shape=(data.shape[0], data.shape[1]))
    lable_placeholder = tf.placeholder(tf.int32, shape=(data.shape[0]))

    input_layer = data_placeholder
    input_size = data.shape[1]
    output_size = data.shape[1]
    hidden_size = 10
    for i in range(hidden_size):
        input_layer = add_layer(input_layer, input_size, output_size, activation_function=tf.nn.relu)

    logits = add_layer(input_layer, input_size, output_size, activation_function=None)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')


    print("")

main()
