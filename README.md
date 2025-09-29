# 使用robert模型（chinese-roberta-wwm-ext-large）在ChnSentiCorp数据集进行微调后，进行情感分析


## 数据集
/dataset/ChnSentiCorp
├── split_dataset
├── load_dataset.py  # 数据加载与划分

## 预训练模型
/models
└── chinese-roberta-wwm-ext-large
    ├── hfl
        └── chinese-roberta-wwm-ext-large
            ├── config.json
            ├── pytorch_model.bin
            └── tokenizer_config.json

## 微调代码
/bert_finetune.py

## 微调后的模型
/models/chinese-roberta-wwm-ext-large_finetune.pth

## API接口
/fastAPI/router.py

## 运行指令
1. 数据集加载与预处理：`python load_dataset.py`
2. 模型训练：`python bert_finetune.py`
3. 启动API服务：`python fastAPI/router.py`
4. 使用curl测试API：`curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "我很开心"}'`




