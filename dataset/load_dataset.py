# #数据集下载
# from modelscope.msdatasets import MsDataset


# LOCAL_DATASET_PATH = 'dataset/ChnSentiCorp'
# ds =  MsDataset.load(LOCAL_DATASET_PATH, subset_name='default', split='train')


# print(f"成功加载本地数据集，样本数量：{len(ds)}")
# print("第一个样本示例：", ds[0])



# git clone https://www.modelscope.cn/datasets/AiNiklaus/ChnSentiCorp.git

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ----------------------------
# 配置参数
# ----------------------------
# 原始CSV文件路径（替换为你的实际路径）
INPUT_CSV_PATH = "/data/icy/code/Bert_Basic/dataset/ChnSentiCorp/ChnSentiCorp_htl_all.csv"
# 划分后的数据保存目录
OUTPUT_DIR = "dataset/ChnSentiCorp/split_dataset"
# 划分比例（测试集20%，验证集占剩余的12.5%，即总数据的10%）
TEST_SIZE = 0.2
VAL_SIZE_FROM_TRAIN_VAL = 0.125  # 0.125 * 0.8 = 0.1（总数据的10%）
RANDOM_STATE = 42  # 固定随机种子，确保结果可复现

# ----------------------------
# 1. 创建输出目录
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 2. 加载原始CSV数据
# ----------------------------
print("加载原始数据集...")
df = pd.read_csv(INPUT_CSV_PATH)
print(f"原始数据总样本数: {len(df)}")
print("原始数据前5行:\n", df.head())

# 检查是否包含必要的字段（假设包含"review"文本列和"label"标签列）
required_columns = ["review", "label"]
if not set(required_columns).issubset(df.columns):
    missing = set(required_columns) - set(df.columns)
    raise ValueError(f"原始CSV缺少必要字段: {missing}")

# ----------------------------
# 3. 划分数据集
# ----------------------------
print("\n开始划分数据集...")

# 第一步：拆分训练+验证集 和 测试集
train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"]  # 按标签分层，保持类别比例
)

# 第二步：从训练+验证集中拆分出训练集和验证集
train_df, val_df = train_test_split(
    train_val_df,
    test_size=VAL_SIZE_FROM_TRAIN_VAL,
    random_state=RANDOM_STATE,
    stratify=train_val_df["label"]
)

# ----------------------------
# 4. 查看划分结果
# ----------------------------
print(f"划分后样本数:")
print(f"训练集: {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"验证集: {len(val_df)} ({len(val_df)/len(df):.1%})")
print(f"测试集: {len(test_df)} ({len(test_df)/len(df):.1%})")

# 查看标签分布是否一致
print("\n标签分布（0=负面，1=正面）:")
print("原始数据:", df["label"].value_counts(normalize=True).round(3))
print("训练集:", train_df["label"].value_counts(normalize=True).round(3))
print("验证集:", val_df["label"].value_counts(normalize=True).round(3))
print("测试集:", test_df["label"].value_counts(normalize=True).round(3))

# ----------------------------
# 5. 保存划分后的数据集
# ----------------------------
train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path = os.path.join(OUTPUT_DIR, "validation.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\n划分后的数据集已保存至: {OUTPUT_DIR}")
print(f"训练集: {train_path}")
print(f"验证集: {val_path}")
print(f"测试集: {test_path}")

