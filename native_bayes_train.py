# -*- coding: utf-8 -*-
import os
import csv
import random
import pickle

import numpy as np
import jieba


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
from models import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

jieba.load_userdict("./data/userdict.txt")


file_path = './data/review.csv'
model_export_path = './data/bayes.pkl'
stopword_path = './data/stopwords.txt'


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
    review = jieba.cut(review)
    all_stop_words = set(stop_words)
    # 去掉停用词
    review_words = [w for w in review if w not in all_stop_words]

    return review_words
def train():
    review_list, sentiment_list = load_corpus(file_path)
    # 将全部的语料按1:4分为测试集与训练集
    n = len(review_list) // 5
    train_review_list, train_sentiment_list = review_list[n:], sentiment_list[n:]
    test_review_list, test_sentiment_list = review_list[:n], sentiment_list[:n]

    print('训练集数量： {}'.format(str(len(train_review_list))))
    print('测试集数量： {}'.format(str(len(test_review_list))))

    review_train = [' '.join(review_to_text(review)) for review in train_review_list]
    sentiment_train = train_sentiment_list

    review_test = [' '.join(review_to_text(review)) for review in test_review_list]
    sentiment_test = test_sentiment_list


    vectorizer = CountVectorizer(max_df=0.8, min_df=3)

    tfidftransformer = TfidfTransformer()

    # 先转换成词频矩阵，再计算TFIDF值
    tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(review_train))
    # 朴素贝叶斯中的多项式分类器
    clf = MultinomialNB().fit(tfidf, sentiment_train)
    # ========== 添加测试集验证 ==========
    print("\n开始测试集验证...")

    # 对测试集进行特征转换
    tfidf_test = tfidftransformer.transform(vectorizer.transform(review_test))

    # 预测测试集
    predictions = clf.predict(tfidf_test)

    # 计算准确率
    accuracy = accuracy_score(sentiment_test, predictions)
    print(f'测试集准确率: {accuracy:.4f}')

    # 打印详细的分类报告
    print("\n分类报告:")
    print(classification_report(sentiment_test, predictions))

    # 打印混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(sentiment_test, predictions)
    print(cm)

    # 计算各类别的准确率
    unique_labels = list(set(sentiment_test))
    print(f"\n各类别详细信息:")
    for label in unique_labels:
        label_indices = [i for i, x in enumerate(sentiment_test) if x == label]
        label_predictions = [predictions[i] for i in label_indices]
        label_accuracy = sum([1 for i, pred in enumerate(label_predictions)
                              if pred == sentiment_test[label_indices[i]]]) / len(label_predictions)
        print(f"类别 '{label}' 准确率: {label_accuracy:.4f}")

    # 展示一些预测错误的样例
    print(f"\n预测错误的样例 (前5个):")
    error_count = 0
    for i, (true_label, pred_label) in enumerate(zip(sentiment_test, predictions)):
        if true_label != pred_label and error_count < 5:
            print(f"样例 {error_count + 1}:")
            print(f"  真实标签: {true_label}")
            print(f"  预测标签: {pred_label}")
            print(f"  文本: {test_review_list[i][:100]}...")
            print("-" * 40)
            error_count += 1
    # ========== 测试集验证结束 ==========

    # 将模型保存pickle文件
    with open(model_export_path, 'wb') as file:
        d = {
            "clf": clf,
            "vectorizer": vectorizer,
            "tfidftransformer": tfidftransformer,
        }
        pickle.dump(d, file)

    print("训练完成")
    return accuracy  # 返回准确率供参考



def test_analyzer():
    """
    测试训练好的朴素贝叶斯分类器
    """
    # 加载训练好的模型
    with open(model_export_path, 'rb') as file:
        model_data = pickle.load(file)
        clf = model_data["clf"]
        vectorizer = model_data["vectorizer"]
        tfidftransformer = model_data["tfidftransformer"]

    def analyze_text(text):
        """
        分析单个文本的情感
        """
        # 预处理文本
        processed_text = ' '.join(review_to_text(text))

        # 转换为特征向量
        X = vectorizer.transform([processed_text])
        X_tfidf = tfidftransformer.transform(X)

        # 预测
        prediction = clf.predict(X_tfidf)[0]
        probabilities = clf.predict_proba(X_tfidf)[0]

        # 获取类别标签
        classes = clf.classes

        # 构建结果
        result = {
            'text': text,
            'prediction': prediction,
            'probabilities': {
                classes[i]: float(probabilities[i])
                for i in range(len(classes))
            }
        }

        return result

    # 测试样例
    test_texts = [
        '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。',
        '这部电影真的太棒了！剧情紧凑，演员演技精湛，特效也很震撼。强烈推荐大家去看！',
        '还可以吧，没有特别惊艳，但也不算太差。中规中矩的一部电影。'
    ]

    print("=" * 60)
    print("朴素贝叶斯情感分析测试结果")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        result = analyze_text(text)
        print(f"\n测试样例 {i}:")
        print(f"文本: {text[:50]}...")
        print(f"预测结果: {result['prediction']}")
        print("概率分布:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
        print("-" * 40)

    return analyze_text


# 在主程序最后添加测试
if __name__ == "__main__":
    print("analyze:::", train())
    # 原有的训练代码...
    # print("训练完成")
    #
    # # 添加测试功能
    analyzer = test_analyzer()

    # 单独测试指定文本
    text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'
    result = analyzer(text)

    print("\n" + "=" * 60)
    print("指定文本分析结果:")
    print("=" * 60)
    print(f"文本: {text}")
    print(f"预测结果: {result['prediction']}")
    print("详细概率:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.6f}")

