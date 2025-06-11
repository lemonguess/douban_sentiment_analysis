import os
import csv
import jieba
import random
import numpy as np
stopword_path = './data/stopwords.txt'
jieba.load_userdict("./data/userdict.txt")
def load_corpus(corpus_path):
    with open(corpus_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    review_data = np.array(rows).tolist()
    # 打乱数据顺序
    random.shuffle(review_data)

    review_list = []
    sentiment_list = []
    for words in review_data:
        review_list.append(words[1])
        sentiment_list.append(words[0])

    return review_list, sentiment_list


def load_stopwords(file_path):
    stop_words = []
    with open(file_path, encoding='UTF-8') as words:
       stop_words.extend([i.strip() for i in words.readlines()])
    return stop_words

# jieba分词
def review_to_text(review):
    stop_words = load_stopwords(stopword_path)
    _review = jieba.cut(review)
    all_stop_words = set(stop_words)
    # 去掉停用词
    review_words = [w for w in _review if w not in all_stop_words]

    return review_words