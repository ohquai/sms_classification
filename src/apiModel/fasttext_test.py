# -*- coding:utf-8 -*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fastText
from fastText import load_model
import os
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import time

file_path = os.path.dirname(os.path.realpath(__file__))
father_path = os.path.dirname(file_path)
data_path = file_path + '/data/'
model_path = file_path + '/model/news_fasttext.bin'
train_path = data_path+'train_news_ttl.txt'
test_path = data_path+'test_news_ttl.txt'
sms_train_path = data_path+'train_sms.txt'
sms_test_path = data_path + 'test_sms.txt'


t1 = time.time()
# load训练好的模型
classifier = fastText.load_model(model_path)
result = classifier.test(sms_test_path, k=1)
print("test file total precesion is: {0}".format(result[1]))
print("test file total recall is:    {0}".format(result[2]))
t2 = time.time()

label_first = False
if label_first:
    label_pos = 0
    text_pos = 1
else:
    label_pos = 1
    text_pos = 0

labels_right = []
texts = []
with open(sms_test_path, encoding='utf8') as fr:
    line = fr.readline()
    while line:
        # print(line)
        # print("#####################")
        # time.sleep(5)
        labels_right.append(line.split("\t")[label_pos].rstrip().replace("__label__", ""))
        texts.append(line.split("\t")[text_pos])
        # labels_right.append(line.split(" ")[0:1].rstrip().replace("__label__", ""))
        # texts.append(" ".join(line.split(" ")[1:]))
        line = fr.readline()
fr.close()

labels_predict = [classifier.predict(text.replace("\n", ""), k=1)[0][0].rstrip().replace("__label__", "") for text in texts]  # 预测输出结果为二维形式
# print(labels_predict)
t3 = time.time()

text_labels = list(set(labels_right))
text_predict_labels = list(set(labels_predict))
print(text_predict_labels)
print(text_labels)

A = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
B = dict.fromkeys(text_labels, 0)  # 测试数据集中各个类的数目
C = dict.fromkeys(text_predict_labels, 0)  # 预测结果中各个类的数目
for i in range(0, len(labels_right)):
    B[labels_right[i]] += 1
    C[labels_predict[i]] += 1
    if labels_right[i] == labels_predict[i]:
        A[labels_right[i]] += 1

# print(A)
# print(B)
# print(C)
# 计算准确率，召回率，F值
for key in B:
    N = int(B[key])
    p = float(A[key]) / float(B[key])
    r = float(A[key]) / float(C[key])
    f = p * r * 2 / (p + r + 0.00001)
    print("[%s]: p->%f  r->%f  f->%f  N->%d" % (key, p, r, f, N))
t4 = time.time()

print("test set predict time %.2fs"%(t2-t1))
print("test set predict time label by label %.2fs"%(t3-t2))
print("test set precision calculate %.2f"%(t4-t3))
