# -*- coding: utf-8 -*-
import gensim
from gensim.models import word2vec
import jieba
import re
import pandas as pd
import numpy as np
import time


class Word2Vec:
    def __init__(self):
        self.source_path = 'D:/Project/fasttext/word2vec/data/risk_sms_sms_cluster_result.csv'
        self.txt_path = 'D:/Project/fasttext/word2vec/data/sms_file.txt'
        self.corpus_path = 'D:/Project/fasttext/word2vec/data/sms_corpus.txt'
        self.model_path = 'D:/Project/fasttext/word2vec/model/sms_word2vec_model'
        self.vector_path = 'D:/Project/fasttext/word2vec/model/sms_word2vec'
        self.stoplist = 'D:/Project/fasttext/word2vec/data/stoplist.txt'

    @staticmethod
    def txt_preprocess(sentence):
        sentence = sentence.strip()
        sentence = sentence.replace("【现金卡】尊敬的", "")
        sentence = sentence.replace("【现金卡】", "")

        p1 = r'^.{2,3}\s{0,1}(，|,)\s{0,1}'  # 去除开头的姓名
        sentence_re = re.sub(p1, "", sentence)

        p2 = r'http:.*。'  # 去除点击的网址
        sentence_re = re.sub(p2, "网址", sentence_re)

        p3 = r'\s{0,1}([1-9]\d{0,9}|0)([.]?|(\.\d{1,2})?)\s{0,1}元'  # 去除金额
        sentence_re = re.sub(p3, "金额", sentence_re)

        p4 = r'\d{4}[年/\\-]\d{1,2}[月/\\-]\d{1,2}日?'  # 去除日期
        sentence_re = re.sub(p4, "日期", sentence_re)

        p5 = r'验证码.{0,3}\d{4,8}'  # 去除验证码
        sentence_re = re.sub(p5, "验证码", sentence_re)

        p6 = r'(客服热线|热线|电话|客服)(\s|:|：)?(\d|-){8,15}'
        sentence_re = re.sub(p6, "电话号", sentence_re)

        # print(sentence)
        # print(sentence_re)
        # time.sleep(1)
        return sentence_re

    def word2vec_generate(self):
        sentences = word2vec.Text8Corpus(self.corpus_path)
        model = word2vec.Word2Vec(sentences, size=50, max_vocab_size=1500, seed=2018, min_count=3, hs=1)
        self.save_model(model)

    def sentence_jieba_split(self):
        stoplist = {}.fromkeys([line.strip() for line in open(self.stoplist, encoding='utf8')])

        with open(self.corpus_path, 'w', encoding='utf8') as fw:
            with open(self.txt_path, 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                for line in lines:
                    line = self.txt_preprocess(line)
                    segs = jieba.cut(line, cut_all=False)
                    segs = [word for word in list(segs) if word not in stoplist]
                    line_seg = " ".join(segs)
                    # line_seg = " ".join(jieba.cut(line))
                    fw.write(line_seg+"\n")
            fr.close()
        fw.close()

    def csv2txt(self):
        df = pd.read_csv(self.source_path, encoding="utf8", names=[1, 2, 3, 4, 5, 6])
        df_content = df.iloc[:, 4]
        with open(self.txt_path, 'w', encoding='utf8') as fw:
            for content in df_content:
                fw.write(content+"\n")
        fw.close()

    def save_model(self, model):
        model.save(self.model_path)
        model.wv.save_word2vec_format(self.vector_path, binary=False)

    def load_model(self):
        new_model = gensim.models.Word2Vec.load(self.model_path)
        return new_model

    def model_evaluate(self):
        model = self.load_model()

        for i in model.most_similar(u"还款"):
            print(i[0], i[1])

        y2 = model.similarity(u"贷款", u"还款")
        print(y2)


if __name__ == "__main__":
    obj = Word2Vec()

    obj.csv2txt()
    obj.sentence_jieba_split()
    obj.word2vec_generate()
    obj.model_evaluate()
