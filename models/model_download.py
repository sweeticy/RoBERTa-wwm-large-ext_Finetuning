from modelscope import snapshot_download

# 指定要下载到的目标路径
target_path = "models/chinese-roberta-wwm-ext-large" 

# 下载模型到指定路径
model_dir = snapshot_download(
    'hfl/chinese-roberta-wwm-ext-large',
    cache_dir=target_path  # 通过cache_dir参数指定下载路径
)

print(f"模型已下载到: {model_dir}")