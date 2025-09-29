import transformers
import os



# 模型实际存储路径（修正为包含权重文件的目录）
model_path = "models/chinese-roberta-wwm-ext-large/hfl/chinese-roberta-wwm-ext-large"


# 实例化bert模型
bert_model = transformers.BertModel.from_pretrained(
    pretrained_model_name_or_path=model_path,
    output_hidden_states=True,  # 配置是否返回所有隐藏层输出（默认只返回最后一层）
    output_attentions=True  # 配置是否返回注意力权重（默认不返回）
)

# bert编码函数
def encoder(model_path, sentence):
    # 步骤1：加载与模型配套的分词器
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)

    # 步骤2：对输入句子进行分词和编码
    tokenized = tokenizer(
        sentence,
        return_tensors='pt'  # 返回的类型为pytorch tensor
    )

    # 步骤3：提取模型需要的三种输入
    input_ids = tokenized['input_ids']  # 分词后的token对应的整数ID
    token_type_ids = tokenized['token_type_ids']    # 区分句子对的标记（单句全为0）
    attention_mask = tokenized['attention_mask']  # 标记有效token（1）和填充（0）
    return input_ids, token_type_ids, attention_mask


# 测试代码
if __name__ == "__main__":
    sentence = "中华人民共和国万岁"
    # 生成三种bert需要的输入形式
    input_ids, token_type_ids, attention_mask = encoder(
        model_path=model_path,
        sentence=sentence
    )
    # 调用bert模型
    sentence_outputs = bert_model(input_ids, token_type_ids, attention_mask)    # BERT 模型对输入文本的特征提取结果
    
    # 输出结果信息
    print("\n模型运行成功!")
    print("最后一层隐藏状态形状:", sentence_outputs.last_hidden_state.shape)
    print("隐藏层数量:", len(sentence_outputs.hidden_states))  # 包括embedding层和所有encoder层
    print("注意力层数量:", len(sentence_outputs.attentions))    # 各个层输出维度(1, num_heads, seq_len,seq_len)）
    



