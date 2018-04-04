# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Embedding
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, Nadam, Adadelta
from keras.initializers import orthogonal, he_normal, he_uniform
import keras.backend as K
from keras.callbacks import Callback, TensorBoard
import tensorflow as tf
from time import sleep
np.random.seed(2018)  # for reproducibility
tf.set_random_seed(2018)

# 设置线程
THREADS_NUM = 3
tf.ConfigProto(intra_op_parallelism_threads=THREADS_NUM)


class FastText():
    """fastText model with Keras, including add pretrained embedding layer weight and self-defined h-softmax"""

    def __init__(self, max_nb_words, max_sequence_length, n_classes, word_index, hidden_layer, lr=0.001):
        self.path = 'D:/Project/fasttext/'
        self.emb_path = self.path + '/model/embedding.bin'
        self.weights_file = self.path + '/model/fasttext.h5'  # download from: http://www.platform.ai/models/

        self.max_nb_words = max_nb_words
        self.max_sequence_length = max_sequence_length
        self.n_classes = n_classes
        self.word_index = word_index
        self.hidden_layer = hidden_layer
        self.lr = lr

        self.model = Sequential()
        self.build_fasttext()
        self.save_model()

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def predict(self, data):
        return self.model.predict(data)

    def build_embedding_weight(self):
        embeddings_index = {}

        f = open(self.emb_path, "rb")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        nb_words = min(self.max_nb_words, len(self.word_index))
        embedding_matrix = np.zeros((nb_words, self.hidden_layer))
        for word, i in self.word_index.items():
            if i >= self.max_nb_words:
                continue
            embedding_vector = embeddings_index.get(str.encode(word))
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def build_fasttext(self):
        """
        可用的initialization方法：random_normal(stddev=0.0001), Orthogonal(), glorot_uniform(), lecun_uniform()
        :return:
        """
        initial_dict = {'orthogonal': orthogonal(), 'he_n': he_normal(), 'he_u': he_uniform()}

        # model.add(Embedding(output_dim=self.hidden_layer, input_length=self.max_sequence_length,
        #                     input_dim=self.max_nb_words, embeddings_initializer=initial_dict['he_n'], trainable=True))
        # model.add(GlobalAveragePooling1D())
        # model.add(Dense(self.n_classes, activation='softmax', kernel_initializer=initial_dict['he_n']))
        self.model.add(Embedding(output_dim=self.hidden_layer, input_length=self.max_sequence_length,
                                 input_dim=self.max_nb_words, trainable=True))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(self.n_classes, activation='softmax'))
        print(self.model.summary())

        self.compile()

    def compile(self):
        self.model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        # self.model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self):
        model = self.model
        model.save(self.weights_file)
