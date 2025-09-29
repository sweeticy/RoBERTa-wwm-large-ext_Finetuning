from flask import Flask, request, jsonify, make_response
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 核心配置：禁用ASCII编码，保留中文
app.config['JSONIFY_MIMETYPE'] = 'application/json;charset=utf-8'  # 显式指定UTF-8编码（可选，增强兼容性）

MODEL_PATH = "chinese_roberta_finetuned/checkpoint-1020"  # 本地模型保存路径
MAX_LENGTH = 512  # 与训练时的max_length一致
device = "cuda" if torch.cuda.is_available() else "cpu"  # 设备

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)  # 替换
# model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(device)  # 替换

model.eval()


@app.route("/", methods=["GET"])
def index():
    # 手动构建响应，确保中文正常显示
    result = {
        "提示": "情感分析API已正常运行",
        "正确使用方式": "发送POST请求到 /predict 接口",
        "请求格式": '{"text": "待预测的中文文本"}',
        "示例（curl）": 'curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d \'{"text": "我很开心"}\''
    }
    # 手动序列化：ensure_ascii=False关闭转义，指定UTF-8编码
    response = make_response(json.dumps(result, ensure_ascii=False))
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.json
    text = data.get("text", "")

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    # 2. 移动输入到模型所在设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. 预测（关闭梯度计算，加速且节省内存）
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. 解析结果
    probs = torch.softmax(outputs.logits, dim=1)  # 转换为概率
    neg_prob = probs[0, 0].item()  # 负面概率
    pos_prob = probs[0, 1].item()  # 正面概率
    pred_label = "正面" if pos_prob > neg_prob else "负面"

    result = {
        "text": text,
        "pred_label": pred_label,
        "neg_prob": round(neg_prob, 6),  # 可选：保留6位小数，更易读
        "pos_prob": round(pos_prob, 6)
    }
    # 手动序列化JSON：ensure_ascii=False关闭中文转义
    response = make_response(json.dumps(result, ensure_ascii=False))
    # 显式指定响应编码为UTF-8，确保客户端正确解析
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)