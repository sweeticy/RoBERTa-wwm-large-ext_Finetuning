import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ----------------------------
# 1. 配置参数
# ----------------------------
MODEL_NAME = "models/chinese-roberta-wwm-ext-large/hfl/chinese-roberta-wwm-ext-large"  # 预训练模型本地路径
MAX_LENGTH = 512  # 文本最大长度
NUM_LABELS = 2  # 分类标签数量（情感分析为2：正/负）
TRAIN_BATCH_SIZE = 8  # 训练批次大小
EVAL_BATCH_SIZE = 8  # 评估批次大小
LEARNING_RATE = 2e-5  # 学习率
NUM_EPOCHS = 3  # 训练轮数
OUTPUT_DIR = "./chinese_roberta_finetuned"  # 模型保存路径
LOG_DIR = "./logs"  # 日志路径

# 本地数据集路径
LOCAL_TRAIN_CSV = "dataset/ChnSentiCorp/split_dataset/train.csv"
LOCAL_VAL_CSV = "dataset/ChnSentiCorp/split_dataset/validation.csv"
LOCAL_TEST_CSV = "dataset/ChnSentiCorp/split_dataset/test.csv"

# 设置设备（优先GPU，没有则用CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


# ----------------------------
# 2. 加载数据集与预处理（核心修改：加载CSV格式+适配字段）
# ----------------------------
def load_and_preprocess_data():
    # 加载本地拆分后的CSV数据集
    dataset = load_dataset(
        "csv",  # 数据集格式为CSV
        data_files={
            "train": LOCAL_TRAIN_CSV,    # 训练集CSV路径
            "validation": LOCAL_VAL_CSV, # 验证集CSV路径
            "test": LOCAL_TEST_CSV       # 测试集CSV路径
        }
    )
    print("数据集结构:", dataset)
    
    # 加载分词器（与模型匹配）
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # 预处理函数：分词、截断、填充
    def preprocess_function(examples):
        # 强制将 review 转为字符串（即使原始是数字/其他类型，避免分词器报错）
        examples["review"] = [str(review).strip() for review in examples["review"]]
        return tokenizer(
            examples["review"],  
            truncation=True,     # 超过MAX_LENGTH自动截断
            padding="max_length",# 不足MAX_LENGTH自动填充
            max_length=MAX_LENGTH
        )
    
    # 应用预处理到整个数据集（批量处理加速）
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    print(f"预处理后训练集样本数: {len(tokenized_dataset['train'])}")  # 训练集样本数
    

    tokenized_dataset = tokenized_dataset.remove_columns(["review"])
    # 重命名标签列为模型要求的"labels"（CSV中标签字段名为"label"）
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    # 转换为PyTorch张量格式（模型直接读取）
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset, tokenizer



# ----------------------------
# 3. 加载模型
# ----------------------------
def load_model():
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    ).to(device)  # 移动模型到指定设备
    return model


# ----------------------------
# 4. 定义评估指标
# ----------------------------
def compute_metrics(eval_pred):
    metric = load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# ----------------------------
# 5. 训练模型
# ----------------------------
def train_model(model, tokenized_dataset):
    # 配置训练参数
    print(f"实际训练批次大小: {TRAIN_BATCH_SIZE}")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        eval_strategy="epoch",  # 每轮评估一次
        save_strategy="epoch",  # 每轮保存一次模型
        logging_dir=LOG_DIR,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,  # 权重衰减（防止过拟合）
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        fp16=torch.cuda.is_available(),  # 若有GPU则启用半精度训练
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()




    return trainer


# ----------------------------
# 6. 评估与预测
# ----------------------------
def predict_with_model(texts, model, tokenizer, device, max_length):
    """
    用指定模型对文本列表进行分类预测
    texts: 待预测的文本列表
    model: 用于预测的模型（预训练模型/微调后模型）
    tokenizer: 对应的分词器
    device: 模型所在设备（GPU/CPU）
    max_length: 文本最大长度
    """
    # 文本预处理（与训练时一致）
    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    # 移动输入到模型所在设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 关闭梯度计算（预测时无需训练，节省资源）
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 解析预测结果（取概率最大的类别）
    probs = torch.softmax(outputs.logits, dim=1)  # 转换为概率
    predictions = torch.argmax(probs, dim=1)      # 取概率最大的类别
    
    # 打印预测结果
    print("\n" + "="*50)
    print("模型预测结果")
    print("="*50)
    for i in range(len(texts)):  # 按样本逐个循环
        text = texts[i]
        # 提取第i个样本的负面/正面概率（此时是单个元素张量，可用.item()）
        neg_prob = probs[i, 0].item()  # 第i个样本，第0类（负面）概率
        pos_prob = probs[i, 1].item()  # 第i个样本，第1类（正面）概率
        # 确定预测类别
        pred_label = "正面" if pos_prob > neg_prob else "负面"
        # 打印结果
        print(f"文本: {text}")
        print(f"预测类别: {pred_label}")
        print(f"负面概率: {neg_prob:.4f}, 正面概率: {pos_prob:.4f}\n")

    return [("正面" if probs[i,1].item() > probs[i,0].item() else "负面") for i in range(len(texts))]


# 在测试集上评估
def evaluate_and_predict(trainer, tokenized_dataset, tokenizer, model):
    
    print("\n在测试集上评估...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"测试集结果: {test_results}")

    


# ----------------------------
# 主函数
# ----------------------------
def main():
    # 加载并预处理数据
    tokenized_dataset, tokenizer = load_and_preprocess_data()
    
    # 加载模型
    model = load_model()
    
    # 训练前测试模型效果
    print("\n微调前预测示例:")
    test_texts = [
        "这家酒店环境优雅，服务周到，下次一定还来！",
        "菜品味道很差，等待时间超长，非常失望。",
        "电影剧情紧凑，演员演技在线，强烈推荐！",
        "这本书内容枯燥，排版混乱，不建议购买。"
    ]
    predict_with_model(test_texts, model, tokenizer, device, MAX_LENGTH)



    # 训练模型
    trainer = train_model(model, tokenized_dataset)



    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Tokenizer已保存到: {OUTPUT_DIR}")
    
    # 评估与预测
    evaluate_and_predict(trainer, tokenized_dataset, tokenizer, model)
    predict_with_model(test_texts, model, tokenizer, device, MAX_LENGTH)


if __name__ == "__main__":
    main()