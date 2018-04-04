# -*- coding:utf-8 -*-
"""
"""
import os
import numpy as np
import pandas as pd
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback, TensorBoard
import tensorflow as tf
from matplotlib import pyplot as plt
from fasttest_model import FastText
# from kerasModel.cnn_model import CNNText
np.random.seed(2018)
tf.set_random_seed(2018)


def data_preprocess(x_train, y_train, x_test, y_test):
    # 训练样本数字化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 为label打上从0开始的标签（不是从零开始的数字会报错）
    # 根据标签的set，为每类标签打上标号
    labels_set = list(set(y_train.tolist()))
    print("we have {0} labels".format(len(labels_set) ))
    print(labels_set)
    n_classes = len(labels_set)
    label_id_dict = {}
    for i in range(n_classes):
        label_id_dict[labels_set[i]] = i

    # 将每个label转换为对应的标号，并转成one-hot格式
    labels_train_category = [label_id_dict[label] for label in y_train]
    labels_train_category = to_categorical(labels_train_category, n_classes)
    labels_val_category = [label_id_dict[label] for label in y_test]
    labels_val_category = to_categorical(labels_val_category, n_classes)

    return x_train, labels_train_category, x_test, labels_val_category, n_classes


def show_model_effect(history):
    # show the data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy and loss")
    # plt.ylabel("accuracy")
    plt.ylabel("indice")
    plt.xlabel("epoch")
    plt.legend(["train_acc", "test_acc"], loc="upper left")

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    # plt.title("Model loss")
    # plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train_loss", "test_loss"], loc="upper right")
    plt.savefig(log_path+"Performance.jpg")


def read_txt(file_path, mode="train"):
    sentence_list = []

    f = open(file_path, encoding='utf8')  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        sentence_list.append(line)
        line = f.readline()
    f.close()
    print("split finished")

    if mode == "train":
        data = pd.DataFrame({"sentence": sentence_list})  # 将字典转换成为数据框
        data["comment"] = data.apply(lambda x: str(x.sentence.split("__label__")[0]), axis=1)
        data["label"] = data.apply(lambda x: x.sentence.split("__label__")[1], axis=1)
        return data["comment"], data["label"].values
    else:
        data = pd.DataFrame({"comment": sentence_list})  # 将字典转换成为数据框
        return data["comment"]


if __name__ == "__main__":
    # paths
    path = 'D:/Project/fasttext/'
    model_path = path + 'model/'
    log_path = path + 'logs/'
    data_path = path + 'data/'
    submission_path = path + 'submissions/sub.csv'
    # train_path = data_path+'news_fasttext_train.txt'
    # test_path = data_path + 'news_fasttext_test.txt'
    # train_path = data_path+'train_news.txt'
    # test_path = data_path + 'test_news.txt'
    train_path = data_path+'train_sms.txt'
    test_path = data_path + 'test_sms.txt'

    VALIDATION_SPLIT = 0.2
    MAX_NB_WORDS = 1000
    MAX_SEQUENCE_LENGTH = 300
    HIDDEN_LAYER = 64
    lr = 0.001
    BATCH_SIZE = 64
    NB_EPOCHS = 25

    # 读取训练和测试数据
    x_train, y_train = read_txt(train_path, mode="train")
    x_test = read_txt(test_path, mode="predict")
    print(u"length of train data is {0}, length of train label is {1}, length of test data is {2}".format(len(x_train), len(y_train), len(x_test)))

    # 生成corpus
    corpus = x_train.tolist() + x_test.tolist()
    print("length of corpus is {0}".format(len(corpus)))

    # 顺序进行tokenizer，fit，txt2sequence，pad-sequence
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(corpus)
    print(u"fit_on_texts finished")
    sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    print("tokenizer finished")
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y_train.shape)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of test_data tensor:', test_data.shape)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    # 将数据的id切分train和val
    perm = np.random.permutation(len(data))
    idx_train = perm[:int(len(data) * (1 - VALIDATION_SPLIT))]
    idx_val = perm[int(len(data) * (1 - VALIDATION_SPLIT)):]

    data_train = data[idx_train]
    labels_train = y_train[idx_train]
    print(data_train.shape, labels_train.shape)

    data_val = data[idx_val]
    labels_val = y_train[idx_val]
    print(data_val.shape, labels_val.shape)

    # 数据预处理
    data_train, labels_train_category, data_val, labels_val_category, n_classes = data_preprocess(data_train, labels_train, data_val, labels_val)

    # 设置fastText一些超参，确定模型结构并编译
    ft = FastText(max_nb_words=MAX_NB_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH, n_classes=n_classes, word_index=word_index, hidden_layer=HIDDEN_LAYER, lr=lr)
    model = ft.model
    model.summary()

    # 训练模型
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = ft.model.fit(x=data_train, y=labels_train_category, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=1, validation_data=(data_val, labels_val_category), callbacks=[early_stopping], shuffle=True)
    show_model_effect(history)
