# -*- coding: utf-8 -*-

import pickle
import numpy as np
import jieba
from models.logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import load_corpus, review_to_text

jieba.load_userdict("./data/userdict.txt")

file_path = './data/review.csv'
model_export_path = './checkpoint/logistic_regression.pkl'
stopword_path = './data/stopwords.txt'


def train():
    review_list, sentiment_list = load_corpus(file_path)
    # 将标签转换为0/1格式
    sentiment_list = [1 if s == '正面' else 0 for s in sentiment_list]

    # 划分训练集/测试集
    n = len(review_list) // 5
    train_reviews, train_labels = review_list[n:], sentiment_list[n:]
    test_reviews, test_labels = review_list[:n], sentiment_list[:n]

    print(f"训练集大小: {len(train_reviews)}")
    print(f"测试集大小: {len(test_reviews)}")
    print("-" * 50)

    # 文本向量化
    vectorizer = CountVectorizer(max_df=0.8, min_df=3)
    tfidftransformer = TfidfTransformer()

    # 处理文本
    processed_reviews = [' '.join(review_to_text(r)) for r in train_reviews]
    X_train = vectorizer.fit_transform(processed_reviews)
    X_train_tfidf = tfidftransformer.fit_transform(X_train)

    print("开始训练逻辑回归模型...")
    print("-" * 50)

    # 训练模型
    model = LogisticRegression(
        learning_rate=0.1,
        num_iterations=20,
        reg_lambda=0.01
    )
    model.fit(X_train_tfidf, train_labels)

    print("-" * 50)
    print("训练完成，开始测试...")

    # 测试集验证
    processed_test = [' '.join(review_to_text(r)) for r in test_reviews]
    X_test = vectorizer.transform(processed_test)
    X_test_tfidf = tfidftransformer.transform(X_test)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(test_labels, predictions)

    # 突出显示测试集准确率
    print("=" * 60)
    print("🎯 测试集评估结果")
    print("=" * 60)
    print(f"📊 测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)


    # 混淆矩阵
    print("\n📈 混淆矩阵:")
    cm = confusion_matrix(test_labels, predictions)
    print(cm)
    print(f"真负例(TN): {cm[0, 0]}, 假正例(FP): {cm[0, 1]}")
    print(f"假负例(FN): {cm[1, 0]}, 真正例(TP): {cm[1, 1]}")

    # 计算其他指标
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📊 其他评估指标:")
    print(f"精确率(Precision): {precision:.4f}")
    print(f"召回率(Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 错误样例展示（限制显示数量）
    print("\n❌ 错误预测样例 (前5个):")
    error_count = 0
    for i, (pred, true, text) in enumerate(zip(predictions, test_labels, test_reviews)):
        if pred != true and error_count < 5:
            error_count += 1
            sentiment_map = {0: '负面', 1: '正面'}
            print(f"样例 {error_count}:")
            print(f"  预测: {sentiment_map[pred[0]]}")
            print(f"  真实: {sentiment_map[true]}")
            print(f"  文本: {text[:80]}...")
            print("-" * 40)

    # 保存模型
    with open(model_export_path, 'wb') as file:
        d = {
            "model": model,
            "vectorizer": vectorizer,
            "tfidftransformer": tfidftransformer,
            "classes": [0, 1],
            "accuracy": accuracy  # 保存准确率
        }
        pickle.dump(d, file)

    print(f"\n💾 模型已保存到: {model_export_path}")
    print(f"🎯 最终测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    return accuracy


def test_analyzer():
    """
    测试训练好的逻辑回归分类器
    """
    print("🔄 正在加载训练好的模型...")

    # 加载训练好的模型
    try:
        with open(model_export_path, 'rb') as file:
            model_data = pickle.load(file)
            clf = model_data["model"]
            vectorizer = model_data["vectorizer"]
            tfidftransformer = model_data["tfidftransformer"]
            classes = model_data["classes"]

            # 如果模型中保存了准确率，显示出来
            if "accuracy" in model_data:
                print(f"📊 模型训练时的测试集准确率: {model_data['accuracy']:.4f} ({model_data['accuracy'] * 100:.2f}%)")

        print("✅ 模型加载成功!")
    except FileNotFoundError:
        print("❌ 模型文件未找到，请先运行训练程序!")
        return None
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    def analyze_text(text):
        """
        分析单个文本的情感
        """
        try:
            # 预处理文本
            processed_text = ' '.join(review_to_text(text))

            # 转换为特征向量
            X = vectorizer.transform([processed_text])
            X_tfidf = tfidftransformer.transform(X)

            # 预测
            prediction = clf.predict(X_tfidf)[0]
            probabilities = clf.predict_proba(X_tfidf)[0]

            # 构建结果
            result = {
                'text': text,
                'prediction': prediction,
                'sentiment': '正面' if prediction == 1 else '负面',
                'confidence': float(max(probabilities))
            }

            return result

        except Exception as e:
            print(f"❌ 文本分析失败: {e}")
            return None

    # 测试样例
    test_texts = [
        '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。',
        '这部电影真的太棒了！剧情紧凑，演员演技精湛，特效也很震撼。强烈推荐大家去看！',
        '还可以吧，没有特别惊艳，但也不算太差。中规中矩的一部电影。',
        '演员表演很自然，故事情节引人入胜，是一部值得观看的好电影。',
        '剧情拖沓，演技尴尬，完全是在浪费时间，不推荐观看。'
    ]

    print("\n" + "=" * 70)
    print("🎭 逻辑回归情感分析测试结果")
    print("=" * 70)

    results = []
    for i, text in enumerate(test_texts, 1):
        result = analyze_text(text)
        if result:
            results.append(result)

            # 根据置信度设置显示样式
            confidence_emoji = "🔥" if result['confidence'] > 0.8 else "👍" if result['confidence'] > 0.6 else "🤔"
            sentiment_emoji = "😊" if result['prediction'] == 1 else "😞"

            print(f"\n📝 测试样例 {i}:")
            print(f"文本: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"预测结果: {sentiment_emoji} {result['sentiment']} {confidence_emoji}")
            print(f"置信度: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
            # print("概率分布:")
            # for label, prob in result['probabilities'].items():
            #     bar_length = int(prob * 20)  # 简单的进度条
            #     bar = "█" * bar_length + "░" * (20 - bar_length)
            #     print(f"  {label}: {prob:.4f} |{bar}| {prob * 100:.2f}%")
            # print("-" * 50)

    # 统计分析结果
    if results:
        positive_count = sum(1 for r in results if r['prediction'] == 1)
        negative_count = len(results) - positive_count
        avg_confidence = sum(r['confidence'] for r in results) / len(results)

        print(f"\n📊 测试样例统计:")
        print(f"正面情感: {positive_count} 个")
        print(f"负面情感: {negative_count} 个")
        print(f"平均置信度: {avg_confidence:.4f} ({avg_confidence * 100:.2f}%)")

    return analyze_text


# 在主程序最后添加测试
if __name__ == "__main__":
    print("🚀 开始训练逻辑回归情感分析模型")
    print("=" * 60)

    # 训练模型并获取准确率
    final_accuracy = train()

    # print("\n" + "=" * 60)
    # print("✅ 训练完成!")
    # print(f"🏆 最终测试集准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    # print("=" * 60)
    #
    # # 可选：运行测试分析器
    # print("\n🔍 运行情感分析测试...")
    # analyzer = test_analyzer()
    #
    # # 单独测试指定文本
    # text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'
    # result = analyzer(text)
    #
    # print("\n" + "=" * 60)
    # print("🎯 指定文本分析结果:")
    # print("=" * 60)
    # print(f"文本: {text}")
    # print(f"预测结果: {'正面' if result['prediction'] == 1 else '负面'}")
