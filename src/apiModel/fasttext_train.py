# -*- coding:utf-8 -*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fastText
from fastText import train_supervised
import os
import time

file_path = os.path.dirname(os.path.realpath(__file__))
father_path = os.path.dirname(file_path)
data_path = file_path + '/data/'
model_path = file_path + '/model/news_fasttext.bin'
train_path = data_path+'train_news_ttl.txt'
test_path = data_path+'test_news_ttl.txt'
sms_train_path = data_path + 'train_sms.txt'
sms_test_path = data_path + 'test_sms.txt'


def print_results(N, p, r):
    print("No. val set is\t" + str(N))
    print("Precis@{}\t{:.3f}".format(1, p))
    # print("Recall@{}\t{:.3f}".format(1, r))


if __name__ == "__main__":
    # 训练模型
    t1 = time.time()
    model = train_supervised(input=sms_train_path, epoch=15, dim=64, lr=0.1, wordNgrams=1, verbose=2, minCount=5, loss="hs")
    t2 = time.time()

    # 测试集上验证
    print_results(*model.test(sms_test_path))
    model.save_model(model_path)
    t3 = time.time()

    # model.quantize(input=train_path, qnorm=True, retrain=True, cutoff=100000)
    # print_results(*model.test(test_path))
    # model.save_model("cooking.ftz")

    print("train time %.2fs"%(t2-t1))
    print("validation time %.2fs"%(t3-t2))
