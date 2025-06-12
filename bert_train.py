from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import load_corpus
from torch.optim import AdamW  # 修改：从torch.optim导入AdamW

# 加载预训练的BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 二分类任务
# 设备指定
# device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# 加载数据集
file_path = './data/review.csv'
review_list, sentiment_list = load_corpus(file_path)

# 将标签转换为数字（0: 负面, 1: 正面）
label_map = {'负面': 0, '正面': 1}
sentiment_list = [int(label) for label in sentiment_list]

# 分割训练集和测试集
train_reviews, test_reviews, train_labels, test_labels = train_test_split(
    review_list, sentiment_list, test_size=0.2, random_state=42
)

# 数据预处理函数
def preprocess_data(reviews, labels, tokenizer, max_length=128):
    inputs = tokenizer(
        reviews,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))

# 准备训练和测试数据
train_dataset = preprocess_data(train_reviews, train_labels, tokenizer)
test_dataset = preprocess_data(test_reviews, test_labels, tokenizer)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练函数
def train(model, train_loader, optimizer, epochs=3):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}")

# 测试函数
def test(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# 主程序
if __name__ == "__main__":
    train(model, train_loader, optimizer)
    test(model, test_loader)
    model_save_path = './checkpoint/bert_sentiment_model'
    # 保存模型和 tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)