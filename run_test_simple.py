
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
    # 原有的训练代码...
    print("训练完成")

    # 添加测试功能
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
