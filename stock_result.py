import tensorflow as tf
import csv
import numpy as np
from tensorflow.python.platform import gfile
import pandas as pd
import math



AIC_PATH = r"D:/project/AIchallenger/data/20171006/ai_challenger_stock_test_20171006/"
AIC_TRAINING = AIC_PATH + "stock_test_data_20171006.csv"
print(AIC_TRAINING)

def stack_csv_load_test_pd(filename,
                   target_dtype,
                   features_dtype):

    data_stock = pd.read_csv(filename)
    data_stock = data_stock.dropna(axis=1)
    data = data_stock.values.astype(features_dtype)
    # target = data_stock.loc[:, "label"].values.astype(target_dtype)
    return {"test":data}


def stack_csv_load_test(filename, target_dtype, features_dtype):
    """Load dataset from CSV file without a header row."""
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data = []
        counts = 0
        for row in data_file:
            counts += 1
            if counts == 1:
                continue
            # if counts == 100:
            #   break
            data.append(np.asarray(row, dtype=features_dtype))

    data = np.array(data)
    return {"test":data}


def data_save_test():
    training_set = stack_csv_load_test_pd(filename=AIC_TRAINING, target_dtype=np.int32, features_dtype=np.float32)
    np.save("test", training_set["test"])

def data_get_test(data):

    test_id = data[:, 0]
    test_data = data[:, 1:-1]
    test_group = data[:, -1]

    print(test_id.shape, test_data.shape, test_group.shape)

    data_set = {"test_id": test_id, "test_data": test_data, "test_group": test_group}
    return data_set


def redef_data(data, split):
    split_below = np.where(data < split)
    split_above = np.where(data >= split)
    data[split_below] = 0.00001
    data[split_above] = 0.99999

    return data


def save_result(id, res, filename='result.csv'):
    f = open(filename, 'w')
    f.write("id,proba\n")
    for i in range(id.shape[0]):
        str = "%d,%.15f\n"% (id[i], res[i])
        f.write(str)
    f.close()


FILE_NAME = "ALL_train_hs2_lr0.00001_mb3000_dk0.08_os87"

def main():
    # data_save_test()
    test = np.load("test.npy")
    dataset = data_get_test(test)

    with tf.device("/cpu:0"):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("ckpt/"+FILE_NAME+".meta")
            saver.restore(sess, "ckpt/"+FILE_NAME)
            softmax_result = tf.get_collection('softmax_result')[0]
            keep_prob = tf.get_collection('keep_prob')[0]
            phase = tf.get_collection('phase')[0]
            X = tf.get_collection('X')[0]
            loss_test = sess.run(softmax_result, feed_dict={X: dataset["test_data"], phase: False, keep_prob: 1})

    id = np.array(dataset["test_id"]).astype(np.int32)
    # id = id.reshape(-1, 1).astype(np.int32)

    # loss_test = redef_data(loss_test, 0.5)
    res = loss_test[:, 1]

    save_result(id, res)
    # result = np.concatenate((id, res), axis=1)
    # np.savetxt('result.csv', result, delimiter=',')

main()

