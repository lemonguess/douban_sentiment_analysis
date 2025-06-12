# -*- coding: utf-8 -*-
import pickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from models.decision_tree import DecisionTree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import load_stopwords, load_corpus, review_to_text

file_path = './data/review.csv'
model_export_path = './checkpoint/decision_tree.pkl'

def train():
    review_list, sentiment_list = load_corpus(file_path)
    # 将全部的语料按1:4分为测试集与训练集
    n = len(review_list) // 5
    train_review_list, train_sentiment_list = review_list[n:], sentiment_list[n:]
    test_review_list, test_sentiment_list = review_list[:n], sentiment_list[:n]

    print('训练集数量： {}'.format(str(len(train_review_list))))
    print('测试集数量： {}'.format(str(len(test_review_list))))

    # 文本预处理
    review_train = [' '.join(review_to_text(review)) for review in train_review_list]
    sentiment_train = train_sentiment_list

    review_test = [' '.join(review_to_text(review)) for review in test_review_list]
    sentiment_test = test_sentiment_list

    # 特征提取
    vectorizer = CountVectorizer(max_df=0.8, min_df=3)
    tfidftransformer = TfidfTransformer()
    tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(review_train))

    # 训练决策树模型
    clf = DecisionTree(max_depth=5, min_samples_split=2, criterion='gini').fit(tfidf.toarray(), sentiment_train)

    # 模型评估
    print("\n开始测试集验证...")
    tfidf_test = tfidftransformer.transform(vectorizer.transform(review_test))
    predictions = clf.predict(tfidf_test.toarray())

    # 计算准确率
    accuracy = accuracy_score(sentiment_test, predictions)
    print(f'测试集准确率: {accuracy:.4f}')

    # 打印详细评估报告
    print("\n分类报告:")
    print(classification_report(sentiment_test, predictions))

    print("\n混淆矩阵:")
    cm = confusion_matrix(sentiment_test, predictions)
    print(cm)

    # 保存模型
    with open(model_export_path, 'wb') as file:
        d = {
            "clf": clf,
            "vectorizer": vectorizer,
            "tfidftransformer": tfidftransformer,
        }
        pickle.dump(d, file)

    print("训练完成")
    return accuracy

if __name__ == "__main__":
    print("决策树训练开始")
    train()